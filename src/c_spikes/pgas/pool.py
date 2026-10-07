"""Opt-in fresh-process PGAS window launcher. The existing serial API is unchanged.

Run ``python -m c_spikes.cli.pgas_pool --help``. Each manifest entry describes one
already-independent window; this module never splits recordings or calibrations.
The coordinator uses only the standard library and never initializes CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def cpu_identity(cpu):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return dict(logical_cpu=cpu, socket=int((topology / "physical_package_id").read_text()),
                core=int((topology / "core_id").read_text()),
                thread_siblings=(topology / "thread_siblings_list").read_text().strip())


@contextmanager
def coordinator_affinity(cpu):
    original = os.sched_getaffinity(0)
    try:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        yield
    finally:
        if cpu is not None:
            os.sched_setaffinity(0, original)


def apply_worker_affinity(task):
    """Pin before importing NumPy/CUDA; all subsequently created threads inherit it."""
    if "worker_cpu" in task:
        cpu = task["worker_cpu"]
        if cpu not in task["allocated_cpus"]:
            raise ValueError("Worker CPU is outside the captured allocation")
        os.sched_setaffinity(0, {cpu})
        if os.sched_getaffinity(0) != {cpu}:
            raise RuntimeError("Worker CPU affinity was not applied")


def cuda_identity():
    """Read the initialized worker's actual CUDA device; never change its policy."""
    import ctypes as ct
    import uuid
    driver = ct.CDLL("libcuda.so.1")
    def call(name, args, types):
        function = getattr(driver, name)
        function.argtypes = types; function.restype = ct.c_int
        code = function(*args)
        if code:
            raise RuntimeError(f"{name} returned CUDA driver error {code}")
    count, device, sm = ct.c_int(), ct.c_int(), ct.c_int()
    flags, memory = ct.c_uint(), ct.c_size_t()
    uid, name = (ct.c_ubyte * 16)(), ct.create_string_buffer(128)
    call("cuDeviceGetCount", [ct.byref(count)], [ct.POINTER(ct.c_int)])
    if count.value != 1:
        raise RuntimeError("Worker must see exactly one CUDA device")
    call("cuCtxGetDevice", [ct.byref(device)], [ct.POINTER(ct.c_int)])
    call("cuDeviceGetUuid_v2", [ct.byref(uid), device], [ct.c_void_p, ct.c_int])
    call("cuDeviceGetName", [name, len(name), device], [ct.c_char_p, ct.c_int, ct.c_int])
    call("cuDeviceTotalMem_v2", [ct.byref(memory), device], [ct.POINTER(ct.c_size_t), ct.c_int])
    call("cuDeviceGetAttribute", [ct.byref(sm), 16, device], [ct.POINTER(ct.c_int), ct.c_int, ct.c_int])
    call("cuCtxGetFlags", [ct.byref(flags)], [ct.POINTER(ct.c_uint)])
    return dict(uuid=str(uuid.UUID(bytes=bytes(uid))), name=name.value.decode(),
                memory_bytes=memory.value, multiprocessors=sm.value,
                context_flags=flags.value, visible_count=count.value)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def stable_seed(fit_id, replicate=0):
    """Positive signed-int CPU seed, independent of worker and execution order."""
    return 1 + int(identity([fit_id, replicate])[:16], 16) % (2**31 - 2)


def verify_files(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f"File identity changed: {path}")


def python_environment():
    from importlib.metadata import PackageNotFoundError, version
    import platform
    result = dict(executable=sys.executable, python_version=platform.python_version())
    for package in ('numpy', 'scipy'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def validate_manifest(manifest):
    if manifest.get('schema_version', 1) not in (1, 2):
        raise ValueError('Unsupported manifest schema')
    if 'python_environment' in manifest and manifest['python_environment'] != python_environment():
        raise ValueError('Pinned Python/dependency environment changed')
    fits = manifest["fits"]
    ids = [fit["fit_id"] for fit in fits]
    if not fits or len(ids) != len(set(ids)):
        raise ValueError("Manifest must have nonempty, unique fit IDs")
    for fit in fits:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", fit["fit_id"]):
            raise ValueError("Unsafe fit ID")
        if fit["seed"] != stable_seed(fit["fit_id"], fit.get("replicate", 0)):
            raise ValueError("Seed does not match fit ID and replicate")
        if (type(fit['config']['niter']) is not int or type(fit['config']['burnin']) is not int
                or not 0 <= fit["config"]["burnin"] < fit["config"]["niter"]):
            raise ValueError("Invalid burn-in")
        for key in ("output_root", "dataset_tag", "constants_file", "gparam_file", "use_cache"):
            if key in fit["config"]:
                raise ValueError(f"Launcher owns config field {key}")
        verify_files(fit["files_sha256"])
        for key in ("input_file", "constants_file", "gparam_file"):
            if fit[key] not in fit["files_sha256"]:
                raise ValueError(f"Unpinned {key}")
    verify_files(manifest["runtime_files_sha256"])
    if manifest["binary"] not in manifest["runtime_files_sha256"]:
        raise ValueError("Unpinned native binary")


def allocation_environment(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("Worker count must be a positive integer")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inference inside a Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len(visible.split(",")) != 1 or visible == "-1":
        raise RuntimeError("Exactly one assigned CUDA device must be visible")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1:
        raise RuntimeError("An explicit Slurm CPU allocation is required")
    env = dict(os.environ)
    env.update({key: "1" for key in THREAD_VARIABLES})
    env.update(C_SPIKES_PGAS_BACKEND="gpu", PYTHONDONTWRITEBYTECODE="1")
    # Keep the parent's verified package first; do not inherit another worktree.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    return env


def task_identity(task):
    """Attempt paths, barriers and assigned core numbers are not scientific identity."""
    if task.get("schema_version") == 2:
        stable = {key: task[key] for key in
                  ("schema_version", "fit", "binary", "runtime_files_sha256", "execution")}
        if 'python_environment' in task:
            stable['python_environment'] = task['python_environment']
        return identity(stable)
    return identity(task)


def verify_completion(attempt, task):
    attempt = Path(attempt)
    receipt = json.loads((attempt / "completion.json").read_text())
    if receipt["task_sha256"] != task_identity(task):
        raise ValueError("Completion belongs to a different task")
    if not receipt.get("artifacts_sha256"):
        raise ValueError("Completion has no verified artifacts")
    if task.get("schema_version") == 2:
        if receipt.get("execution") != task["execution"] or receipt.get("seed") != task["fit"]["seed"]:
            raise ValueError("Completion modes or seed differ from frozen task")
        mode = task['execution']['mps']
        if mode != 'unverified':
            for stamp in ('mps_start', 'mps_end'):
                check = receipt[stamp]
                if not check['verified'] or check['driver_mps_enabled'] != int(mode == 'require'):
                    raise ValueError('Completion lacks verified MPS mode')
                if mode == 'require':
                    from c_spikes.pgas.mps import pids
                    if receipt['pid'] not in pids(check['service']['client_list']['stdout']):
                        raise ValueError('Completion lacks actual client membership')
            if mode == 'require':
                barrier = receipt['mps_barrier']
                if not barrier['released'] or receipt['pid'] not in barrier['pids']:
                    raise ValueError('Completion lacks verified startup admission')
    for name, digest in receipt["artifacts_sha256"].items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt.resolve()) or sha256(path) != digest:
            raise ValueError(f"Completion artifact changed: {name}")
    return receipt


def attempts_for(root, fit_id):
    return sorted((Path(root) / fit_id).glob("attempt-[0-9][0-9][0-9][0-9]"))


def previous_result(root, task, resume=False, retry_failed=False):
    attempts = attempts_for(root, task['fit']['fit_id'])
    if not attempts:
        return None
    last = attempts[-1]
    try:
        previous = json.loads((last / 'task.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        if retry_failed:
            return None  # Frozen root identity was checked before this preflight.
        raise FileExistsError('Partial task record; use --retry-failed')
    if task_identity(previous) != task_identity(task):
        raise ValueError('Existing fit directory has a different identity')
    # A child can publish a receipt and then crash, or the coordinator can die.
    # Neither case has a committed successful process record and is reusable.
    try:
        process = json.loads((last / 'process.json').read_text())
        successful = (process['status'] == 'completed' and process['returncode'] == 0
                      and process.get('task_sha256') == task_identity(task))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        successful = False
    if successful:
        if not resume:
            raise FileExistsError('Completed fit exists; use --resume')
        try:
            receipt = verify_completion(last, task)
        except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError):
            if retry_failed:
                return None
            raise
        return dict(fit_id=task['fit']['fit_id'], status='cached', receipt=receipt, output=str(last))
    if not retry_failed:
        raise FileExistsError('Partial/failed fit; use --retry-failed for a fresh attempt')
    return None


def run_one(task, root, env, resume=False, retry_failed=False, command=None, children=None):
    """One fresh process; preserve failure evidence and never automatically retry."""
    from c_spikes.pgas.queue import Children
    children = children or Children()
    previous = previous_result(root, task, resume, retry_failed)
    if previous is not None:
        return previous
    fit_id = task['fit']['fit_id']
    fit_dir = Path(root) / fit_id
    fit_dir.mkdir(exist_ok=True)
    attempts = attempts_for(root, fit_id)
    number = int(attempts[-1].name.split('-')[1]) + 1 if attempts else 1
    if number > 9999:
        raise RuntimeError('Attempt number limit reached')
    attempt = fit_dir / f'attempt-{number:04d}'
    attempt.mkdir()
    atomic_json(attempt / 'task.json', task)
    argv = command or [sys.executable, '-m', 'c_spikes.cli.pgas_pool', '--fit', str(attempt / 'task.json')]
    record = dict(fit_id=fit_id, output=str(attempt), returncode=None,
                  task_sha256=task_identity(task), status='failed')
    started = time.monotonic()
    process = None
    try:
        if children.stop.is_set():
            raise InterruptedError('Queue cancelled before worker startup')
        with (attempt / 'worker.log').open('w') as log:
            process = subprocess.Popen(argv, cwd=attempt, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=() if children.lock_fd is None else (children.lock_fd,))
            children.register(fit_id, process)
            atomic_json(attempt / 'process-start.json', dict(pid=process.pid, argv=argv,
                        job_id=env.get('SLURM_JOB_ID'), task_sha256=task_identity(task)))
            while process.poll() is None:
                if children.stop.wait(.05):
                    children.terminate(process)
                    raise InterruptedError('Queue cancelled; owned worker group terminated')
            record['returncode'] = process.returncode
            if process.returncode:
                raise RuntimeError(f'Worker exited {process.returncode}')
            record.update(status='completed', receipt=verify_completion(attempt, task))
    except Exception as exc:
        record.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
        if process is not None:
            children.terminate(process)
            record['returncode'] = process.returncode
    finally:
        children.unregister(fit_id)
        record['process_wall_s'] = time.monotonic() - started
        if record['status'] != 'completed':
            atomic_json(attempt / 'failure.json', record)
        atomic_json(attempt / 'process.json', record)
    return record


def run_manifest(manifest, root, workers=None, resume=False, retry_failed=False,
                 worker_cpus=None, coordinator_cpu=None, mps_expected=None, **options):
    from c_spikes.pgas.queue import run_queue
    return run_queue(manifest, root, workers, resume, retry_failed,
                     worker_cpus, coordinator_cpu, mps_expected, **options)


def fit_worker(task_file):
    """Always leave actionable evidence, including failures before native import."""
    out = Path(task_file).resolve().parent
    try:
        _fit_worker(task_file)
    except BaseException as exc:
        import traceback
        from c_spikes.pgas.mps_gate import environment, diagnostic_snapshot
        record = dict(pid=os.getpid(), error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc(), environment=environment(),
                      cpu_affinity=sorted(os.sched_getaffinity(0)))
        try:
            task = json.loads(Path(task_file).read_text())
            if 'mps_expected' in task and not (out/'mps-start.json').exists():
                diagnostic_snapshot(task['mps_expected'], out/'mps-start.json')
        except Exception as diagnostic_error:
            record['diagnostic_error'] = str(diagnostic_error)
        atomic_json(out/'worker-error.json', record)
        raise


def _fit_worker(task_file):
    """Import native code only in this fresh interpreter, after thread limits."""
    import importlib.util
    import resource
    import types
    started = time.monotonic()
    task_file = Path(task_file).resolve()
    out = task_file.parent
    task = json.loads(task_file.read_text())
    apply_worker_affinity(task)
    fit = task["fit"]
    if 'mps_expected' in task:
        from c_spikes.pgas.mps_gate import diagnostic_snapshot, environment, wait_for_release
        atomic_json(out/'execution-start.json', dict(pid=os.getpid(), stage='before native import',
                    environment=environment(), cpu_affinity_start=sorted(os.sched_getaffinity(0))))
        if task['mps_expected'] and task.get('schema_version') == 2:
            from c_spikes.pgas.mps_gate import scoped_pipe
            scoped_pipe()  # Never initialize CUDA against an unverified service.
    verify_files(task["runtime_files_sha256"])
    verify_files(fit["files_sha256"])
    if task.get('python_environment') is not None and task['python_environment'] != python_environment():
        raise ValueError('Pinned Python/dependency environment changed')
    import numpy as np
    native_spec = importlib.util.spec_from_file_location("pgas_bound_gpu", task["binary"])
    native = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native)
    execution = {}
    if "worker_cpu" in task or "mps_expected" in task:
        execution = dict(pid=os.getpid(), cpu_identity=cpu_identity(task["worker_cpu"]) if "worker_cpu" in task else None,
                         cpu_affinity_start=sorted(os.sched_getaffinity(0)))
        if 'mps_expected' in task:
            execution['environment'] = environment()
        try:
            execution['cuda_device'] = cuda_identity()
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
            uuid_mismatch = (visible.startswith(("MIG-", "GPU-"))
                             and execution["cuda_device"]["uuid"] != visible.split("-", 1)[1])
            if uuid_mismatch or execution["cuda_device"]["context_flags"] & 7:
                raise RuntimeError("Unexpected GPU identity or nondefault CUDA scheduling flags")
            if "mps_expected" in task:
                execution['environment'] = environment()
                execution['mps_start'] = diagnostic_snapshot(task['mps_expected'], out/'mps-start.json')
        except Exception as exc:
            execution['error'] = str(exc)
            if 'mps_expected' in task and not (out/'mps-start.json').exists():
                try: diagnostic_snapshot(task['mps_expected'], out/'mps-start.json')
                except Exception as diagnostic_error: execution['diagnostic_error'] = str(diagnostic_error)
            raise
        finally:
            atomic_json(out / "execution-start.json", execution)
        if 'mps_barrier' in task:
            execution['mps_barrier'] = wait_for_release(task['mps_barrier'], fit['fit_id'],
                dict(pid=os.getpid(), mps=execution['mps_start'], cuda_device=execution['cuda_device'],
                     cpu_affinity=execution['cpu_affinity_start']))
    native_wall = []
    class TimedAnalyzer:
        def __init__(self, **kwargs):
            self.inner = native.Analyzer(**kwargs)
        def run(self):
            t = time.monotonic()
            self.inner.run()
            native_wall.append(time.monotonic() - t)
    sys.modules["c_spikes.pgas.pgas_bound"] = types.SimpleNamespace(Analyzer=TimedAnalyzer)
    from c_spikes.inference.pgas import PgasConfig, run_pgas_inference
    from c_spikes.inference.cache import set_cache_root
    from c_spikes.inference.types import TrialSeries, ensure_serializable
    imported = time.monotonic()
    with np.load(fit["input_file"], allow_pickle=False) as data:
        times = np.ascontiguousarray(data["time"], dtype=np.float64)
        values = np.ascontiguousarray(data["fluorescence"], dtype=np.float64)
    if times.shape != values.shape or times.ndim != 1 or len(times) != fit["n_frames"]:
        raise ValueError("Input dimensions changed")
    if not np.isfinite(times).all() or not np.isfinite(values).all() or not (np.diff(times) > 0).all():
        raise ValueError("Invalid window")
    constants = json.loads(Path(fit["constants_file"]).read_text())
    constants["MCMC"]["seed"] = fit["seed"]
    if 'seeded_constants' in fit and constants != fit['seeded_constants']:
        raise ValueError('Seeded constants differ from frozen manifest')
    atomic_json(out / "constants.json", constants)
    set_cache_root(out / "cache")
    config = PgasConfig(dataset_tag=fit["fit_id"], output_root=out / "raw",
                        constants_file=out / "constants.json", gparam_file=Path(fit["gparam_file"]),
                        use_cache=False, **fit["config"])
    loaded = time.monotonic()
    result = run_pgas_inference([TrialSeries(times=times, values=values)],
                                fit["raw_fs"], np.zeros(0, dtype=np.float64), config)
    inferred = time.monotonic()
    if "mps_expected" in task:
        execution["mps_end"] = diagnostic_snapshot(task["mps_expected"], out/'mps-end.json')
    if result.spike_prob.shape != result.time_stamps.shape or not np.isfinite(result.spike_prob).all():
        raise ValueError("Invalid result")
    np.savez(out / "result.npz", time=result.time_stamps, spikes=result.spike_prob,
             reconstruction=result.reconstruction, discrete_spikes=result.discrete_spikes)
    atomic_json(out / "metadata.json", ensure_serializable(result.metadata))
    artifacts = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*")
                 if p.is_file() and p.name not in ("worker.log", "task.json")}
    verify_files(task["runtime_files_sha256"])
    verify_files(fit["files_sha256"])
    if "worker_cpu" in task:
        thread_affinity = {}
        for path in Path("/proc/self/task").iterdir():
            try:
                thread_affinity[path.name] = sorted(os.sched_getaffinity(int(path.name)))
            except ProcessLookupError:
                pass  # A runtime helper thread can exit between enumeration and query.
        if any(cpus != [task["worker_cpu"]] for cpus in thread_affinity.values()):
            raise RuntimeError("A worker thread escaped its requested core")
        execution["thread_affinity_end"] = thread_affinity
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt = dict(task_sha256=task_identity(task), fit_id=fit["fit_id"], seed=fit["seed"],
                   gpu_seed=42, gpu_seed_lifecycle="reset each sweep (unchanged main)",
                   artifacts_sha256=artifacts, cache_used=False,
                   import_s=imported-started, input_s=loaded-imported,
                   pipeline_s=inferred-loaded, native_s=sum(native_wall),
                   output_s=time.monotonic()-inferred, total_s=time.monotonic()-started,
                   max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime,
                   job_id=os.environ.get("SLURM_JOB_ID"), hostname=os.uname().nodename,
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   cpu_affinity=sorted(os.sched_getaffinity(0)), python=sys.executable,
                   numpy_version=np.__version__, native_path=native.__file__)
    receipt['python_environment'] = python_environment()
    if "execution" in task:
        receipt["execution"] = task["execution"]
    receipt.update(execution)
    atomic_json(out / "completion.json", receipt)
