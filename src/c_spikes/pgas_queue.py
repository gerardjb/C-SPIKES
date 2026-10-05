"""Bounded Linux subprocess scheduling; the coordinator never initializes CUDA."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time


class Children:
    """Own only process groups created by this invocation, including on cancellation."""

    def __init__(self, stop=None, lock_fd=None):
        self.stop = stop if stop is not None else threading.Event()
        self.lock_fd = lock_fd
        self.processes = {}
        self.lock = threading.Lock()

    def register(self, label, process):
        with self.lock:
            self.processes[label] = process

    def unregister(self, label):
        with self.lock:
            self.processes.pop(label, None)

    def pids(self):
        with self.lock:
            return {label: p.pid for label, p in self.processes.items() if p.poll() is None}

    @staticmethod
    def terminate(process):
        # start_new_session=True makes this an owned group, not the coordinator's.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    @contextmanager
    def signals(self):
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda *_: self.stop.set())
        try:
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def execution_policy(workers, mps_mode, placement):
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if mps_mode not in ('off', 'require', 'unverified'):
        raise ValueError('MPS mode must be off or require')
    if placement not in ('shared', 'separate', 'inherit', 'explicit'):
        raise ValueError('Unknown CPU placement')
    from c_spikes.pgas_pool import THREAD_VARIABLES
    return dict(workers=workers, backend='gpu', mps=mps_mode,
                cpu_placement=placement, cuda_wait_policy='default',
                thread_limits={name: '1' for name in THREAD_VARIABLES},
                cpu_seed_policy='sha256-fit-id-replicate-v1',
                gpu_seed=42, gpu_seed_lifecycle='reset each sweep (unchanged native)')


def cpu_slots(workers, placement, allocated, requested=None):
    from c_spikes.pgas_pool import cpu_identity
    cores = {}
    for cpu in allocated:
        detail = cpu_identity(cpu)
        cores.setdefault((detail['socket'], detail['core']), cpu)
    if requested is not None:
        if len(requested) != workers or any(c not in allocated for c in requested):
            raise ValueError('Worker CPU slots must remain inside the allocation and match workers')
        slots = list(requested)
    elif placement == 'separate':
        if len(cores) < workers:
            raise ValueError('Separate placement needs one allocated physical core per worker')
        slots = list(cores.values())[:workers]
    elif placement == 'shared':
        slots = [allocated[0]] * workers
    else:
        slots = [None] * workers
    if placement == 'separate':
        identities = [cpu_identity(cpu) for cpu in slots]
        if len({(c['socket'], c['core']) for c in identities}) != workers:
            raise ValueError('Separate placement cannot share a physical core or SMT siblings')
    if placement == 'shared' and len(set(slots)) != 1:
        raise ValueError('Shared placement requires the same core for all slots')
    return slots


def run_queue(manifest, root, workers=None, resume=False, retry_failed=False,
              worker_cpus=None, coordinator_cpu=None, mps_expected=None,
              *, mps_mode=None, cpu_placement=None, slot_cpus=None,
              service_file=None, stop_event=None, startup_timeout=60):
    from c_spikes import pgas_pool as pool
    import fcntl
    pool.validate_manifest(manifest)
    frozen_policy = manifest.get('execution', {})
    if workers is None:
        workers = frozen_policy.get('workers', 1)
    env = pool.allocation_environment(workers)
    allocated = sorted(os.sched_getaffinity(0))
    if mps_expected is not None and type(mps_expected) is not bool:
        raise ValueError('MPS expectation must be boolean')
    mode = mps_mode or ('require' if mps_expected else 'off' if mps_expected is False
                       else frozen_policy.get('mps', 'unverified'))
    placement = cpu_placement or ('explicit' if worker_cpus is not None
                                  else frozen_policy.get('cpu_placement', 'inherit'))
    policy = execution_policy(workers, mode, placement)
    if worker_cpus is not None:
        if (set(worker_cpus) != {f['fit_id'] for f in manifest['fits']}
                or any(c not in allocated for c in worker_cpus.values())
                or coordinator_cpu not in allocated):
            raise ValueError('CPU plan must cover every fit and remain inside the allocation')
        policy['worker_cpus'] = worker_cpus
        policy['coordinator_cpu'] = coordinator_cpu
    elif coordinator_cpu is not None and coordinator_cpu not in allocated:
        raise ValueError('Coordinator CPU is outside the allocation')
    slots = cpu_slots(workers, placement, allocated, slot_cpus)
    selected = set(worker_cpus.values()) if worker_cpus else {c for c in slots if c is not None}
    if len(selected) > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('CPU placement exceeds Slurm cpus-per-task')
    if placement != 'inherit' and coordinator_cpu is None:
        coordinator_cpu = slots[0]
    if mode == 'require' and placement == 'inherit':
        raise ValueError('MPS verification requires explicit CPU placement')
    if 'execution' in manifest and manifest['execution'] != policy:
        raise ValueError('Requested execution modes differ from frozen manifest')
    frozen_manifest = dict(manifest, execution=policy)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = root / 'manifest.json'
        if frozen.exists() and pool.identity(json.loads(frozen.read_text())) != pool.identity(frozen_manifest):
            raise ValueError('Output root belongs to another manifest')
        pool.atomic_json(frozen, frozen_manifest)
        tasks = [dict(schema_version=2, fit=fit, binary=manifest['binary'],
                      runtime_files_sha256=manifest['runtime_files_sha256'], execution=policy,
                      python_environment=manifest.get('python_environment'))
                 for fit in manifest['fits']]
        records = [pool.previous_result(root, task, resume, retry_failed) for task in tasks]
        pending = [i for i, record in enumerate(records) if record is None]
        # No service or device access is needed for an entirely cached queue.
        run = root / ('run-' + secrets.token_hex(8))
        run.mkdir()
        children = Children(stop_event, lock.fileno())
        env['C_SPIKES_MPS_ALLOCATED_CPUS'] = ','.join(map(str, allocated))
        service_file = service_file or env.get('C_SPIKES_MPS_SERVICE_FILE')
        if mode == 'require' and pending:
            if not service_file:
                raise ValueError('MPS requires a current allocation service scope file; see site setup')
            scope = json.loads(Path(service_file).read_text())
            env['C_SPIKES_MPS_SERVICE_FILE'] = str(Path(service_file).resolve())
            env['CUDA_MPS_PIPE_DIRECTORY'] = scope['pipe_directory']
            env.pop('C_SPIKES_MPS_OFF_DIRECTORY', None)
        elif mode == 'off':
            # Never shut down a service. An empty directory prevents auto-attachment.
            env.pop('C_SPIKES_MPS_SERVICE_FILE', None)
            private = run / 'mps-off'
            private.mkdir(mode=0o700)
            env['CUDA_MPS_PIPE_DIRECTORY'] = str(private)
            env['C_SPIKES_MPS_OFF_DIRECTORY'] = str(private)
        started = time.monotonic()
        gates = []
        futures = {}
        error = None

        def dispatch(executor, indices, free_slots):
            barrier = None
            if mode == 'require':
                directory = run / ('gate-' + secrets.token_hex(8)); directory.mkdir()
                barrier = dict(directory=str(directory), token=secrets.token_hex(16),
                               participants=[tasks[i]['fit']['fit_id'] for i in indices],
                               timeout_s=startup_timeout)
            new = []
            for index, slot in zip(indices, free_slots):
                task = dict(tasks[index], allocated_cpus=allocated, run_directory=str(run))
                cpu = worker_cpus[task['fit']['fit_id']] if worker_cpus else slots[slot]
                if cpu is not None:
                    task['worker_cpu'] = cpu
                if mode != 'unverified':
                    task['mps_expected'] = mode == 'require'
                if barrier:
                    task['mps_barrier'] = barrier
                future = executor.submit(pool.run_one, task, root, env, resume, retry_failed,
                                         children=children)
                futures[future] = (index, slot)
                new.append(future)
            if barrier:
                from c_spikes.mps_gate import release_when_verified
                decision = release_when_verified(barrier,
                    lambda: children.stop.is_set() or any(f.done() for f in new),
                    expected_pids=children.pids, env=env)
                gates.append(decision)
                if not decision['released']:
                    # A broken service/gate is a queue-wide failure, not a retry loop.
                    children.stop.set()

        with children.signals(), pool.coordinator_affinity(coordinator_cpu):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                try:
                    initial = pending[:workers]; pending = pending[workers:]
                    if initial and not children.stop.is_set():
                        dispatch(executor, initial, range(len(initial)))
                    else:
                        pending = initial + pending
                    while futures:
                        done, _ = wait(futures, timeout=.05, return_when=FIRST_COMPLETED)
                        free = []
                        for future in done:
                            index, slot = futures.pop(future)
                            try:
                                records[index] = future.result()
                            except Exception as exc:
                                records[index] = dict(fit_id=tasks[index]['fit']['fit_id'],
                                                     status='failed', error=str(exc))
                                raise
                            free.append(slot)
                        if pending and free and not children.stop.is_set():
                            indices = pending[:len(free)]; pending = pending[len(free):]
                            dispatch(executor, indices, sorted(free))
                except BaseException as exc:
                    children.stop.set()
                    error = f'{type(exc).__name__}: {exc}'
                finally:
                    for future, (index, _) in futures.items():
                        try:
                            records[index] = future.result()
                        except Exception as exc:
                            records[index] = dict(fit_id=tasks[index]['fit']['fit_id'],
                                                  status='failed', error=str(exc))
        for i, record in enumerate(records):
            if record is None:
                records[i] = dict(fit_id=tasks[i]['fit']['fit_id'], status='cancelled',
                                  error='Queue stopped before dispatch')
        report = dict(manifest_sha256=pool.identity(frozen_manifest), execution=policy,
                      workers=workers, fits=records, batch_wall_s=time.monotonic()-started,
                      job_id=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                      thread_limits={k: env[k] for k in pool.THREAD_VARIABLES},
                      cpu_plan=dict(allocated_cpus=allocated, slot_cpus=slots,
                                    worker_cpus=worker_cpus, coordinator_cpu=coordinator_cpu),
                      startup_gates=gates, cancelled=children.stop.is_set(), error=error)
        if gates:
            report['mps_startup_gate'] = gates[0]  # Compatibility for a one-wave benchmark.
        for key, status in [('new_successes', 'completed'), ('failures', 'failed'),
                            ('cached', 'cached'), ('cancelled_fits', 'cancelled')]:
            report[key] = sum(r['status'] == status for r in records)
        pool.atomic_json(run / 'batch.json', report)
        pool.atomic_json(root / 'batch.json', report)
        # Only remove the empty off directory created by this invocation.
        if mode == 'off':
            try:
                private.rmdir()
            except OSError:
                pass  # Preserve unexpected content for diagnosis; never delete a service.
        return report
