"""Read-only allocation-scoped MPS evidence; never starts or stops a service."""
import ctypes as ct
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import stat
import subprocess


def private_pipe():
    if os.environ.get('C_SPIKES_MPS_SERVICE_FILE'):
        from c_spikes.mps_gate import scoped_pipe
        return scoped_pipe()
    path = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"])
    off = os.environ.get('C_SPIKES_MPS_OFF_DIRECTORY')
    if off and path == Path(off) and path.is_absolute() and not path.is_symlink():
        info = path.stat()
        if info.st_uid == os.getuid() and stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700:
            if any(path.iterdir()):
                raise RuntimeError('Ordinary-CUDA directory is not empty')
            return path
        raise RuntimeError('Ordinary-CUDA directory must be private and owned by this user')
    expected = f"cspikes-mps-{os.getuid()}-{os.environ['SLURM_JOB_ID']}"
    if path.parent != Path('/tmp') or path.name != expected or path.is_symlink():
        raise RuntimeError("MPS pipe is not the allocation's private directory")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError("MPS pipe directory must be owned by this user with mode 0700")
    return path


def control(command, required=True):
    private_pipe()
    result = subprocess.run(['nvidia-cuda-mps-control'], input=command+'\n',
                            text=True, capture_output=True, timeout=5)
    row = dict(command=command, returncode=result.returncode,
               stdout=result.stdout, stderr=result.stderr)
    if required and result.returncode:
        raise RuntimeError(f"MPS control failed: {row}")
    return row


def pids(text):
    # Reject headings, device ordinals or partial/sub-string matches as client evidence.
    return [int(line.strip()) for line in text.splitlines() if re.fullmatch(r'\s*[1-9][0-9]*\s*',line)]


def process_identity(pid):
    root = Path('/proc')/str(pid)
    if root.stat().st_uid != os.getuid():
        raise RuntimeError("MPS service is not owned by this allocation's user")
    cgroup = (root/'cgroup').read_text()
    if not re.search(r"job[_-]"+re.escape(os.environ['SLURM_JOB_ID'])+r"(?:/|$)",cgroup,re.M):
        raise RuntimeError("MPS service is outside the current job cgroup")
    allowed = {int(c) for c in os.environ['C_SPIKES_MPS_ALLOCATED_CPUS'].split(',')}
    masks = {}
    for task in (root/'task').iterdir():
        try:
            mask = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
        if not set(mask) <= allowed:
            raise RuntimeError("MPS service thread escaped the allocation CPUs")
        masks[task.name] = mask
    env = dict(x.split('=',1) for x in (root/'environ').read_bytes().decode().split('\0') if '=' in x)
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    pipe_value = env.get('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')
    pipe = private_pipe()
    if os.environ.get('C_SPIKES_MPS_SERVICE_FILE'):
        import json
        from c_spikes.mps_gate import verify_scope
        scope=json.loads(Path(os.environ['C_SPIKES_MPS_SERVICE_FILE']).read_text())
        verify_scope(dict(uid=os.getuid(),cgroup=cgroup,environment=env,thread_affinity=masks),
                     visible,allowed,pipe,scope.get('device_scope'))
    elif env.get('CUDA_VISIBLE_DEVICES') != visible or pipe_value != str(pipe):
        raise RuntimeError("MPS service has a different device or communication directory")
    return dict(pid=pid, uid=os.getuid(), cgroup=cgroup, thread_affinity=masks,
                command=(root/'cmdline').read_bytes().decode().replace('\0',' ').strip(),
                environment={k:v for k,v in env.items() if k.startswith('CUDA_')})


def service_snapshot(required_clients):
    daemon_pid = int((private_pipe()/'nvidia-cuda-mps-control.pid').read_text().strip())
    servers = control('get_server_list')
    listing = {pid:control(f'get_client_list {pid}') for pid in pids(servers['stdout'])}
    matching = [pid for pid,row in listing.items() if set(required_clients) <= set(pids(row['stdout']))]
    if len(matching) != 1:
        raise RuntimeError(f"Real inference PIDs {required_clients} are not attached to one MPS server: {listing}")
    pid = matching[0]
    client_scope = {}
    if os.environ.get('C_SPIKES_MPS_SERVICE_FILE'):
        # A site-default endpoint is acceptable only when every attached client
        # also belongs to this allocation; sharing the same Unix user is insufficient.
        for client in pids(listing[pid]['stdout']):
            try:
                client_scope[client] = process_identity(client)
            except (FileNotFoundError, ProcessLookupError):
                # Unrelated workers may finish between listing and /proc lookup.
                # Required clients must remain live until their own gate releases.
                again = control(f'get_client_list {pid}')
                if client in required_clients or client in pids(again['stdout']):
                    raise
    return dict(utc=datetime.now(timezone.utc).isoformat(), required_clients=required_clients,
                server_pid=pid, server_list=servers, client_list=listing[pid],
                client_scope=client_scope,
                device_client_list=control(f'get_device_client_list {pid}',required=False),
                ps=control(f'ps -p {pid}',required=False),
                daemon=process_identity(daemon_pid), server=process_identity(pid))


def cuda_mps_enabled():
    driver = ct.CDLL('libcuda.so.1')
    dev, value = ct.c_int(), ct.c_int()
    function = driver.cuCtxGetDevice
    function.argtypes=[ct.POINTER(ct.c_int)];function.restype=ct.c_int
    if function(ct.byref(dev)):
        raise RuntimeError('No current CUDA context for MPS verification')
    function = driver.cuDeviceGetAttribute
    function.argtypes=[ct.POINTER(ct.c_int),ct.c_int,ct.c_int];function.restype=ct.c_int
    if function(ct.byref(value),133,dev):  # CU_DEVICE_ATTRIBUTE_MPS_ENABLED, CUDA 12.9 header
        raise RuntimeError('CUDA MPS-enabled attribute unavailable')
    return value.value


def worker_snapshot(expected):
    pipe = private_pipe()
    enabled = cuda_mps_enabled()
    if enabled != int(expected):
        raise RuntimeError(f"MPS expected={expected}, driver attribute={enabled}; reject ordinary-CUDA fallback")
    record = dict(pid=os.getpid(), expected=expected, driver_mps_enabled=enabled,
                  environment={k:v for k,v in os.environ.items() if k.startswith(('CUDA_','OMP_','OPENBLAS_','MKL_'))})
    if expected:
        record['service'] = service_snapshot([os.getpid()])
    elif (pipe/'nvidia-cuda-mps-control.pid').exists():
        raise RuntimeError('Unexpected service in the ordinary-CUDA control directory')
    return record
