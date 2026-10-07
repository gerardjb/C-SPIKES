"""Allocation-scoped MPS discovery, durable diagnostics and a pre-inference gate.

No service startup or GPU reconfiguration. All CUDA work belongs to the caller.
"""
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time

from c_spikes.pgas.pool import atomic_json


def environment():
    prefixes = ('CUDA_', 'OMP_', 'OPENBLAS_', 'MKL_', 'C_SPIKES_MPS_')
    slurm = {'SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_PARTITION', 'SLURM_JOB_NODELIST',
             'SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
             '_SLURM_SPANK_OPTION_nvidia_spank_gpu_mps', 'SPANK__SLURM_SPANK_OPTION_nvidia_spank_gpu_mps'}
    return {k: v for k, v in os.environ.items() if k.startswith(prefixes) or k in slurm}


def in_job(cgroup, job):
    return bool(re.search(r'job[_-]' + re.escape(job) + r'(?:/|$)', cgroup, re.M))


def process_record(pid):
    root = Path('/proc') / str(pid)
    record = dict(pid=pid, uid=root.stat().st_uid)
    if record['uid'] != os.getuid():
        raise RuntimeError('Process is owned by another user')
    record['cgroup'] = (root / 'cgroup').read_text()
    if not in_job(record['cgroup'], os.environ['SLURM_JOB_ID']):
        raise RuntimeError('Process is outside this Slurm job')
    record['argv'] = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    record['start_ticks'] = (root / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    raw = dict(x.split('=', 1) for x in (root / 'environ').read_bytes().decode().split('\0') if '=' in x)
    record['environment'] = {k: v for k, v in raw.items() if k.startswith('CUDA_')
                             or k in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_GPUS')}
    record['thread_affinity'] = {}
    for task in (root / 'task').iterdir():
        try:
            record['thread_affinity'][task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
    return record


def verify_scope(record, assigned_uuid, cpus, pipe, device_scope=None):
    if record['uid'] != os.getuid() or not in_job(record['cgroup'], os.environ['SLURM_JOB_ID']):
        raise RuntimeError('Service ownership/job scope mismatch')
    visible = record['environment'].get('CUDA_VISIBLE_DEVICES')
    if visible != assigned_uuid:
        if (visible is not None or not device_scope or not device_scope.get('verified')
                or device_scope['cgroup'] != record['cgroup']
                or device_scope['device_uuids'] != [assigned_uuid]):
            raise RuntimeError('Service GPU scope lacks explicit UUID or matching cgroup visibility proof')
    if record['environment'].get('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps') != str(pipe):
        raise RuntimeError('Service/client communication directory mismatch')
    if not record['thread_affinity'] or any(not mask or not set(mask) <= set(cpus)
                                          for mask in record['thread_affinity'].values()):
        raise RuntimeError('Service thread affinity exceeds allocated CPUs')


def endpoint(pipe, daemon_pid):
    """Only an endpoint whose PID file belongs to the current job's daemon."""
    pipe = Path(pipe)
    if not pipe.is_absolute() or pipe.is_symlink():
        raise RuntimeError('Invalid service directory')
    info = pipe.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError('Service directory is not owned by this user')
    pidfile = pipe / 'nvidia-cuda-mps-control.pid'
    if pidfile.is_symlink() or pidfile.stat().st_uid != os.getuid():
        raise RuntimeError('Untrusted service PID file')
    if int(pidfile.read_text().strip()) != daemon_pid:
        raise RuntimeError('Endpoint belongs to a different daemon')
    process_record(daemon_pid)  # Recheck job ownership before each query.
    return pipe


def query(command, pipe, daemon_pid):
    record = dict(command=command, returncode=None, stdout='', stderr='')
    try:
        endpoint(pipe, daemon_pid)
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(pipe))
        result = subprocess.run(['nvidia-cuda-mps-control'], input=command+'\n',
                                text=True, capture_output=True, timeout=5, env=env)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        def text(value): return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        record.update(error='control query timed out', stdout=text(exc.stdout), stderr=text(exc.stderr))
    except Exception as exc:
        record['error'] = str(exc)
    return record


def capture_logs(daemon, target):
    """Copy bounded tails of this user's service logs; preserve failures explicitly."""
    target = Path(target); target.mkdir(exist_ok=True)
    directory = Path(daemon['environment'].get('CUDA_MPS_LOG_DIRECTORY', '/var/log/nvidia-mps'))
    rows = []
    for name in ('control.log', 'server.log'):
        path = directory / name
        row = dict(path=str(path), available=False)
        try:
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                raise RuntimeError('Log is not owned by this user or is a symlink')
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                content = stream.read(65536)
            saved = target / name; saved.write_bytes(content)
            row.update(available=True, saved=str(saved), bytes=len(content), note='At most final 64 KiB; may contain earlier entries in a site-default log')
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)
    return rows


def discover_service(target, assigned_uuid, cpus):
    """Inspect only same-user, current-job services before clients initialize CUDA."""
    report = dict(verified=False, assigned_uuid=assigned_uuid, allocated_cpus=cpus,
                  job_id=os.environ['SLURM_JOB_ID'], environment=environment(), candidates=[], errors=[])
    try:
        for root in Path('/proc').iterdir():
            if not root.name.isdigit():
                continue
            try:
                if root.stat().st_uid != os.getuid():
                    continue
                if not in_job((root/'cgroup').read_text(), os.environ['SLURM_JOB_ID']):
                    continue
                argv = (root/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                if not argv or Path(argv[0]).name != 'nvidia-cuda-mps-control' or '-d' not in argv:
                    continue
                row = process_record(int(root.name)); report['candidates'].append(row)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except Exception as exc:
                report['errors'].append(dict(pid=root.name, error=str(exc)))
        if len(report['candidates']) != 1:
            raise RuntimeError('Expected exactly one MPS daemon owned by this Slurm job')
        daemon = report['candidates'][0]
        pipe = daemon['environment'].get('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')
        report.update(daemon=daemon, pipe_directory=pipe)
        # Queries are read-only and only contact a daemon with verified job ownership.
        report['server_list'] = query('get_server_list', pipe, daemon['pid'])
        report['logs'] = capture_logs(daemon, Path(target)/'service-logs-start')
        if daemon['environment'].get('CUDA_VISIBLE_DEVICES') is None:
            report['device_scope'] = device_scope_probe(target, daemon, assigned_uuid)
        verify_scope(daemon, assigned_uuid, cpus, pipe, report.get('device_scope'))
        endpoint(pipe, daemon['pid'])
        if report['server_list']['returncode'] != 0:
            raise RuntimeError('Scoped MPS control query failed')
        report['verified'] = True
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        if 'server_list' not in report:
            report['server_list'] = dict(command='get_server_list', returncode=None,
                stdout='', stderr='', executed=False,
                reason='No verified current-job daemon endpoint; do not query a shared/default service')
        atomic_json(Path(target)/'service-scope.json', report)
    return report


def device_scope_probe(target, daemon, assigned_uuid):
    """Measure device filtering in the exact service cgroup; never create a context."""
    record=dict(verified=False)
    try:
        config=Path('/etc/slurm/cgroup.conf').read_text()
        record['cgroup_configuration']=config
        record['cgroup']=Path('/proc/self/cgroup').read_text()
        if not re.search(r'^ConstrainDevices\s*=\s*yes\s*$',config,re.M|re.I):
            raise RuntimeError('Compute node does not enable device cgroups')
        if record['cgroup']!=daemon['cgroup']:
            raise RuntimeError('Enumeration process and service are not in the identical cgroup')
        with tempfile.TemporaryDirectory(prefix=f"cspikes-enum-{os.getuid()}-") as private:
            env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=private)
            env.pop('CUDA_VISIBLE_DEVICES',None)
            output=Path(target)/'device-scope-enumeration.json'
            result=subprocess.run([sys.executable,'-m','c_spikes.pgas.mps_device_scope',str(output)],
                                  env=env,text=True,capture_output=True,timeout=15)
            record['enumeration_process']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            enumeration=json.loads(output.read_text());record['enumeration']=enumeration
            record['device_uuids']=[d['uuid'] for d in enumeration['devices']]
            if (result.returncode or not enumeration['completed']
                    or enumeration['cgroup']!=daemon['cgroup']
                    or enumeration['cuda_visible_devices'] is not None
                    or enumeration['driver_api_version']<12080
                    or record['device_uuids']!=[assigned_uuid]):
                raise RuntimeError('Unfiltered CUDA enumeration did not prove exactly the assigned slice')
            record['verified']=True
    except Exception as exc:record['error']=str(exc)
    return record


def scoped_pipe():
    scope = json.loads(Path(os.environ['C_SPIKES_MPS_SERVICE_FILE']).read_text())
    if not scope['verified'] or scope['job_id'] != os.environ['SLURM_JOB_ID']:
        raise RuntimeError('No verified current-job MPS scope')
    allowed = {int(c) for c in os.environ['C_SPIKES_MPS_ALLOCATED_CPUS'].split(',')}
    if set(scope['allocated_cpus']) != allowed:
        raise RuntimeError('Service scope and current allocation CPUs differ')
    pipe = Path(scope['pipe_directory'])
    daemon = process_record(scope['daemon']['pid'])
    if daemon['start_ticks'] != scope['daemon']['start_ticks']:
        raise RuntimeError('MPS daemon PID was reused')
    verify_scope(daemon, os.environ['CUDA_VISIBLE_DEVICES'], scope['allocated_cpus'], pipe, scope.get('device_scope'))
    if os.environ.get('CUDA_MPS_PIPE_DIRECTORY') != str(pipe):
        raise RuntimeError('Worker targets a different communication directory')
    return endpoint(pipe, daemon['pid'])


def diagnostic_snapshot(expected, path):
    """Write driver, environment and control evidence before any rejection."""
    from c_spikes.pgas import mps
    record = dict(pid=os.getpid(), expected=expected, environment=environment(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), verified=False, queries=[])
    try:
        try:
            record['driver_mps_enabled'] = mps.cuda_mps_enabled()
        except Exception as exc:
            record['driver_error'] = str(exc)
        # Continue gathering evidence even when the driver says ordinary CUDA.
        pipe = mps.private_pipe()
        pidfile = pipe/'nvidia-cuda-mps-control.pid'
        record['pipe_directory'] = str(pipe)
        record['daemon_pid_file_present'] = pidfile.exists()
        if expected:
            daemon_pid = int(pidfile.read_text().strip()) if pidfile.exists() else None
            if daemon_pid is not None:
                record['queries'].append(query('get_server_list', pipe, daemon_pid))
                for pid in mps.pids(record['queries'][0]['stdout']):
                    record['queries'].append(query(f'get_client_list {pid}', pipe, daemon_pid))
            else:
                # A verified private, empty directory cannot target another service.
                row = dict(command='get_server_list', returncode=None, stdout='', stderr='')
                record['queries'].append(row)
                try:
                    result = subprocess.run(['nvidia-cuda-mps-control'], input='get_server_list\n',
                                            text=True, capture_output=True, timeout=5)
                    row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                except subprocess.TimeoutExpired as exc:
                    text = lambda value: value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                    row.update(error='control query timed out', stdout=text(exc.stdout), stderr=text(exc.stderr))
            if record.get('driver_mps_enabled') != 1:
                raise RuntimeError('MPS required but driver attribute is not 1; reject ordinary-CUDA fallback')
            record['service'] = mps.service_snapshot([os.getpid()])
        elif record.get('driver_mps_enabled') != 0 or pidfile.exists():
            raise RuntimeError('Ordinary-CUDA control has unexpected MPS state')
        record['verified'] = True
    except Exception as exc:
        record['error'] = str(exc)
        if os.environ.get('C_SPIKES_MPS_SERVICE_FILE'):
            try:
                scoped_pipe()
                scope = json.loads(Path(os.environ['C_SPIKES_MPS_SERVICE_FILE']).read_text())
                record['service_logs'] = capture_logs(scope['daemon'], Path(path).parent/'mps-failure-logs')
            except Exception as log_error:
                record['service_logs_error'] = str(log_error)
    finally:
        atomic_json(path, record)
    if not record['verified']:
        raise RuntimeError(record['error'])
    return record


def wait_for_release(barrier, label, ready):
    root = Path(barrier['directory'])
    atomic_json(root/(label+'.json'), dict(ready, token=barrier['token'], label=label))
    deadline = time.monotonic()+barrier['timeout_s']
    while time.monotonic() < deadline:
        if (root/'decision.json').exists():
            decision = json.loads((root/'decision.json').read_text())
            if decision.get('token') != barrier['token']:
                raise RuntimeError('Startup barrier identity mismatch')
            if not decision.get('released'):
                raise RuntimeError('Startup barrier rejected: '+decision.get('error', 'unknown'))
            if ready['pid'] not in decision['pids']:
                raise RuntimeError('Worker absent from verified startup barrier')
            return decision
        time.sleep(.05)
    raise RuntimeError('Startup barrier timed out; inference not started')


def snapshot_in_environment(ids, env, target):
    """Keep allocation-specific environment out of coordinator globals/threads."""
    output = Path(target) / 'service-check.json'
    result = subprocess.run([sys.executable, '-m', 'c_spikes.pgas.mps_gate', str(output),
                             *map(str, ids)], env=env, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'Service verification failed: {result.stderr}; see {output}')
    return json.loads(output.read_text())


def release_when_verified(barrier, failed, expected_pids=None, env=None):
    """All participants stay alive/idle while one simultaneous membership is checked."""
    from c_spikes.pgas.mps import service_snapshot
    root = Path(barrier['directory'])
    decision = dict(token=barrier['token'], released=False)
    try:
        deadline = time.monotonic()+barrier['timeout_s']
        while time.monotonic() < deadline:
            if failed():
                raise RuntimeError('A worker exited before all participants passed attachment')
            paths = [root/(label+'.json') for label in barrier['participants']]
            if all(p.exists() for p in paths):
                ready = [json.loads(p.read_text()) for p in paths]
                if any(r['token'] != barrier['token'] or not r['mps']['verified']
                       or r['mps']['driver_mps_enabled'] != 1 for r in ready):
                    raise RuntimeError('A participant has invalid attachment evidence')
                ids = [r['pid'] for r in ready]
                if len(set(ids)) != len(ids):
                    raise RuntimeError('Duplicate client PIDs')
                if expected_pids is not None:
                    actual = expected_pids()
                    if any(actual.get(r['label']) != r['pid'] for r in ready):
                        raise RuntimeError('Barrier PID does not identify the launched live worker')
                service = (service_snapshot(ids) if env is None else
                           snapshot_in_environment(ids, env, root))
                decision.update(service=service, pids=ids, ready=ready, released=True)
                return decision
            time.sleep(.05)
        raise RuntimeError('Participants did not all arrive before the startup deadline')
    except Exception as exc:
        decision['error'] = str(exc)
        return decision
    finally:
        atomic_json(root/'decision.json', decision)


if __name__ == '__main__':
    from c_spikes.pgas.mps import service_snapshot
    try:
        atomic_json(Path(sys.argv[1]), service_snapshot([int(pid) for pid in sys.argv[2:]]))
    except Exception as exc:
        atomic_json(Path(sys.argv[1]), dict(error=str(exc), environment=environment()))
        raise
