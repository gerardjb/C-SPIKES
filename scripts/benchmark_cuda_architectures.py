"""Compare two installed CUDA builds in isolated, warmed worker processes.

Run on Linux with an otherwise idle GPU. Supply a NumPy .npz containing 1-D
``time`` and ``data`` arrays, and the same constants/GCaMP files for both builds.
Only Analyzer.run() and its final CUDA synchronization are timed. Imports,
input preparation, and output checks are excluded; native initialization,
parameter updates, and trajectory file writes within run() are included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time


def worker(args):
    # Keep the control channel separate from native C++ stdout messages.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import ctypes
    import numpy as np
    from c_spikes.pgas import pgas_bound_gpu as pgas

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuCtxSynchronize.restype = ctypes.c_int

    def synchronize():
        status = cuda.cuCtxSynchronize()
        if status != 0:
            raise RuntimeError(f"cuCtxSynchronize failed: {status}")

    source = np.load(args.input)
    times = np.asarray(source["time"], dtype=np.float64)
    data = np.asarray(source["data"], dtype=np.float64)
    assert times.ndim == data.ndim == 1 and times.shape == data.shape
    assert np.isfinite(times).all() and np.isfinite(data).all()
    assert (np.diff(times) > 0).all()
    base = json.loads(args.constants.read_text())
    protocol.write(json.dumps({"ready": True, "module": pgas.__file__,
                               "module_sha256": hashlib.sha256(Path(pgas.__file__).read_bytes()).hexdigest(),
                               "python": sys.version, "numpy": np.__version__}) + "\n")
    for line in sys.stdin:
        request = json.loads(line)
        case = request["case"]
        out = args.output / request["run_id"]
        out.mkdir(parents=True, exist_ok=False)
        constants = json.loads(json.dumps(base))
        constants["data"]["sampling_frequency"] = float(1 / np.median(np.diff(times)))
        constants["MCMC"].update(nparticles=case["particles"], niter=args.iterations,
                                 seed=17, SAMPLE_PARAMETERS=True, KNOWN_SPIKES=False)
        config = out / "constants.json"
        config.write_text(json.dumps(constants, indent=2) + "\n")
        os.environ["C_SPIKES_PGAS_RESAMPLING"] = case["resampling"]
        os.environ["C_SPIKES_PGAS_TRAJECTORY"] = case["trajectory"]
        if case["resampling"] == "device":
            os.environ["C_SPIKES_PGAS_ANCESTOR_SEED"] = "17"
        else:
            os.environ.pop("C_SPIKES_PGAS_ANCESTOR_SEED", None)
        model = pgas.Analyzer(time=times, data=data, constants_file=str(config),
                              output_folder=str(out), column=0, tag="bench",
                              niter=args.iterations, trainedPriorFile="", append=False,
                              trim=1, verbose=False, gtSpikes=np.array([0.]),
                              has_trained_priors=False, has_gtspikes=False, maxlen=0,
                              Gparam_file=str(args.gparams), seed=17)
        synchronize()
        started_at = time.time()
        start = time.perf_counter()
        model.run()
        synchronize()
        seconds = time.perf_counter() - start
        if model.get_run_status() != "completed":
            raise RuntimeError(model.get_run_message())
        params = model.get_parameter_samples()
        trajectory_path = out / "traj_samples_bench.dat"
        trajectory = np.loadtxt(trajectory_path, delimiter=",", skiprows=1)
        assert params.shape == (args.iterations, 6) and np.isfinite(params).all()
        assert trajectory.shape == (args.iterations * len(times), 6)
        assert np.isfinite(trajectory).all()
        response = {**request, "started_at_unix": started_at, "seconds": seconds,
                    "parameter_shape": list(params.shape),
                    "trajectory_shape": list(trajectory.shape), "finite": True,
                    "parameter_sha256": hashlib.sha256(params.tobytes()).hexdigest(),
                    "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest()}
        (out / "result.json").write_text(json.dumps(response, indent=2) + "\n")
        protocol.write(json.dumps(response) + "\n")
        del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-python", type=Path)
    parser.add_argument("--hopper-python", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--gparams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("input", "constants", "gparams", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        worker(args)
        return
    if not args.portable_python or not args.hopper_python:
        parser.error("Provide both installed Python environments")
    if min(args.iterations, args.repeats, args.warmups) < 1:
        parser.error("Iterations, repeats, and warmups must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_DISABLE_PTX_JIT="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_PROC_BIND="false", C_SPIKES_PGAS_BACKEND="gpu")
    env.pop("CUDA_FORCE_PTX_JIT", None)
    env.pop("PYTHONPATH", None)
    cases = [
        {"name": "default_1000", "particles": 1000, "resampling": "host", "trajectory": "full"},
        {"name": "device_1000", "particles": 1000, "resampling": "device", "trajectory": "selected"},
        {"name": "device_8192", "particles": 8192, "resampling": "device", "trajectory": "selected"},
    ]
    report = {"iterations": args.iterations, "repeats": args.repeats,
              "warmups_per_case_per_build": args.warmups, "cases": cases,
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "constants_sha256": hashlib.sha256(args.constants.read_bytes()).hexdigest(),
              "gparams_sha256": hashlib.sha256(args.gparams.read_bytes()).hexdigest(),
              "environment": {key: value for key, value in env.items()
                              if key.startswith(("CUDA_", "OMP_", "OPENBLAS_", "MKL_"))},
              "workers": {}, "runs": [], "summary": {}}
    workers = {}
    logs = []

    def response(name):
        if not select.select([workers[name].stdout], [], [], 300)[0]:
            raise TimeoutError(f"{name} worker exceeded 300 seconds")
        line = workers[name].stdout.readline()
        if not line:
            raise RuntimeError(f"{name} worker stopped; see {args.output / (name + '.log')}")
        return json.loads(line)

    try:
        for name, python in [("portable", args.portable_python), ("hopper", args.hopper_python)]:
            log = (args.output / f"{name}.log").open("w")
            logs.append(log)
            # Resolving a venv's Python symlink would select the base interpreter.
            command = [str(python.absolute()), str(Path(__file__).resolve()), "--worker",
                       "--input", str(args.input), "--constants", str(args.constants),
                       "--gparams", str(args.gparams), "--output", str(args.output / name),
                       "--iterations", str(args.iterations)]
            p = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=log, text=True, bufsize=1, cwd=args.output)
            workers[name] = p
            report["workers"][name] = response(name)
        for case in cases:
            for phase, repeats in [("warmup", args.warmups), ("measured", args.repeats)]:
                for repeat in range(repeats):
                    order = ("portable", "hopper") if repeat % 2 == 0 else ("hopper", "portable")
                    for name in order:
                        request = {"case": case, "phase": phase, "repeat": repeat,
                                   "variant": name, "run_id": f"{case['name']}_{phase}_{repeat}"}
                        p = workers[name]
                        p.stdin.write(json.dumps(request) + "\n")
                        p.stdin.flush()
                        row = response(name)
                        report["runs"].append(row)
                        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(f"{case['name']} {phase} {repeat} {name}: {row['seconds']:.6f}s", flush=True)
            measured = [row for row in report["runs"] if row["phase"] == "measured"
                        and row["case"]["name"] == case["name"]]
            times = {name: [row["seconds"] for row in measured if row["variant"] == name]
                     for name in workers}
            ratios = [p / h for p, h in zip(times["portable"], times["hopper"])]
            report["summary"][case["name"]] = {
                "seconds": times,
                "median_seconds": {name: statistics.median(values) for name, values in times.items()},
                "paired_portable_over_hopper": ratios,
                "median_paired_speedup": statistics.median(ratios),
            }
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
    finally:
        for p in workers.values():
            if p.poll() is None:
                p.stdin.close()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
