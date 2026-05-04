#!/usr/bin/env python3
"""Code Ocean smoke test for GPU visibility and one-epoch inference.

This stage is intentionally small: it records accelerator/backend visibility,
then runs the standard batch inference CLI on one selected jGCaMP8f epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATASET = "jGCaMP8f_ANM478349_cell04"
DEFAULT_TRIAL = 1
DEFAULT_METHODS = ("pgas", "ens2", "cascade")
RUN_TAG = "code_ocean_smoke_single_epoch"


def _env_flag(name: str, default: bool) -> bool:
    token = os.environ.get(name)
    if token is None:
        return default
    return token.strip().lower() not in {"0", "false", "no", "off"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=os.environ.get("C_SPIKES_SMOKE_DATASET", DEFAULT_DATASET))
    parser.add_argument(
        "--trial",
        type=int,
        default=int(os.environ.get("C_SPIKES_SMOKE_TRIAL", str(DEFAULT_TRIAL))),
        help="0-based trial index to process from the selected dataset.",
    )
    parser.add_argument(
        "--methods",
        default=os.environ.get("C_SPIKES_SMOKE_METHODS", ",".join(DEFAULT_METHODS)),
        help="Comma-separated methods to run. Default: pgas,ens2,cascade.",
    )
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("C_SPIKES_SMOKE_REQUIRE_GPU", True),
        help="Fail if no GPU is visible to torch/tensorflow/nvidia-smi.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        default=_env_flag("C_SPIKES_SMOKE_PLAN_ONLY", False),
        help="Write selection/probe artifacts without launching inference.",
    )
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_capture(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            text=True,
            capture_output=True,
            env=dict(env) if env is not None else None,
        )
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc), "cmd": list(cmd)}
    return {
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": list(cmd),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _python_probe(code: str, *, env: Mapping[str, str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        capture_output=True,
        env=dict(env),
    )
    payload: dict[str, Any] = {
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    if proc.stdout.strip().startswith("{"):
        try:
            payload["json"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return payload


def _base_env(repo_root: Path, scratch_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    if env.get("C_SPIKES_USE_SOURCE_TREE") == "1":
        src = str(repo_root / "code" / "src")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(scratch_dir / "mpl_cache"))
    env.setdefault("XDG_CACHE_HOME", str(scratch_dir / "mpl_cache"))
    env.setdefault("PYTHONPYCACHEPREFIX", str(scratch_dir / "python_cache"))
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("C_SPIKES_TF_SUPPRESS_RUNTIME_STDERR", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def _probe_hardware(env: Mapping[str, str]) -> dict[str, Any]:
    torch_code = r"""
import json
import torch
out = {
    "version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
print(json.dumps(out, sort_keys=True))
"""
    tf_code = r"""
import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
out = {
    "version": tf.__version__,
    "gpu_count": len(gpus),
    "gpus": [gpu.name for gpu in gpus],
}
print(json.dumps(out, sort_keys=True))
"""
    pgas_code = r"""
import json
import c_spikes.pgas.pgas_bound as pgas_bound
out = {
    "backend": getattr(pgas_bound, "__backend__", None),
    "has_analyzer": hasattr(pgas_bound, "Analyzer"),
}
print(json.dumps(out, sort_keys=True))
"""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "nvidia_smi": _run_capture(["nvidia-smi", "-L"], env=env),
        "torch": _python_probe(torch_code, env=env),
        "tensorflow": _python_probe(tf_code, env=env),
        "pgas": _python_probe(pgas_code, env=env),
    }


def _gpu_visible(probe: Mapping[str, Any]) -> bool:
    nvidia_ok = bool(probe.get("nvidia_smi", {}).get("available"))
    torch_json = probe.get("torch", {}).get("json", {})
    tf_json = probe.get("tensorflow", {}).get("json", {})
    torch_ok = bool(torch_json.get("cuda_available")) and int(torch_json.get("device_count", 0)) > 0
    tf_ok = int(tf_json.get("gpu_count", 0)) > 0
    return nvidia_ok and (torch_ok or tf_ok)


def _parse_methods(token: str) -> list[str]:
    methods = []
    for part in token.replace(";", ",").split(","):
        method = part.strip().lower()
        if method == "all":
            method = ""
            for default in DEFAULT_METHODS:
                if default not in methods:
                    methods.append(default)
        elif method:
            if method not in {"pgas", "ens2", "cascade"}:
                raise ValueError(f"Unknown smoke method: {method}")
            if method not in methods:
                methods.append(method)
    return methods or list(DEFAULT_METHODS)


def _run_inference(
    *,
    repo_root: Path,
    data_dir: Path,
    results_dir: Path,
    env: Mapping[str, str],
    dataset: str,
    trial: int,
    methods: Sequence[str],
    plan_only: bool,
) -> tuple[Path, float]:
    selection = results_dir / "smoke_trial_selection.json"
    _write_json(selection, {dataset: [int(trial)]})

    output_root = results_dir / "full_evaluation"
    cache_root = results_dir / "inference_cache"
    pgas_output_root = results_dir / "pgas_output"
    cmd: list[str] = [
        sys.executable,
        "-m",
        "c_spikes.cli.run",
        "--data-root",
        str(data_dir / "sample_data" / "janelia_8f" / "excitatory"),
        "--dataset",
        dataset,
        "--edges-path",
        str(data_dir / "reference_inputs" / "edges" / "excitatory_time_stamp_edges.npy"),
        "--trial-selection-path",
        str(selection),
        "--cache-root",
        str(cache_root),
        "--output-root",
        str(output_root),
        "--pgas-output-root",
        str(pgas_output_root),
        "--pgas-constants",
        str(data_dir / "parameter_files" / "constants_GCaMP8_soma.json"),
        "--pgas-gparam",
        str(repo_root / "code" / "src" / "c_spikes" / "pgas" / "20230525_gold.dat"),
        "--pgas-bm-sigma",
        "0.03",
        "--ens2-pretrained-root",
        str(data_dir / "Pretrained_models" / "ENS2" / "ens2_published"),
        "--cascade-model-root",
        str(data_dir / "Pretrained_models" / "CASCADE"),
        "--cascade-model-name",
        "universal_p_cascade_exc_30",
        "--cascade-no-discrete",
        "--smoothing-level",
        "raw",
        "--corr-sigma-ms",
        "50",
        "--trialwise-correlations",
        "--run-tag",
        RUN_TAG,
    ]
    for method in methods:
        cmd.extend(["--method", method])

    plan = {
        "dataset": dataset,
        "trial": int(trial),
        "methods": list(methods),
        "command": cmd,
        "summary_path": str(output_root / RUN_TAG / dataset / "raw" / "summary.json"),
    }
    _write_json(results_dir / "smoke_plan.json", plan)
    print("[smoke] " + " ".join(cmd), flush=True)
    start = time.perf_counter()
    if not plan_only:
        subprocess.run(cmd, cwd=str(repo_root), env=dict(env), check=True)
    walltime_s = time.perf_counter() - start
    return output_root / RUN_TAG / dataset / "raw" / "summary.json", walltime_s


def _validate_summary(summary_path: Path, methods: Sequence[str], *, plan_only: bool) -> dict[str, Any]:
    if plan_only:
        return {"plan_only": True, "summary_path": str(summary_path)}
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods_run = set(summary.get("methods_run", []))
    missing = sorted(set(methods) - methods_run)
    if missing:
        raise RuntimeError(f"Smoke summary missing methods {missing}: {summary_path}")
    return {
        "plan_only": False,
        "summary_path": str(summary_path),
        "methods_run": sorted(methods_run),
        "correlations": summary.get("correlations", {}),
        "gt_count": summary.get("gt_count"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    smoke_start = time.perf_counter()
    args = parse_args(argv)
    repo_root = _repo_root()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    scratch_dir = args.scratch_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    methods = _parse_methods(args.methods)
    env = _base_env(repo_root, scratch_dir)
    if args.require_gpu and "pgas" in methods:
        env["C_SPIKES_PGAS_BACKEND"] = "gpu"

    probe_start = time.perf_counter()
    probe = _probe_hardware(env)
    probe_walltime_s = time.perf_counter() - probe_start
    _write_json(results_dir / "gpu_environment.json", probe)
    if args.require_gpu and not _gpu_visible(probe):
        raise RuntimeError(
            "GPU smoke test requested, but no GPU was visible. "
            f"See {results_dir / 'gpu_environment.json'}."
        )
    pgas_json = probe.get("pgas", {}).get("json", {})
    if "pgas" in methods and args.require_gpu and pgas_json.get("backend") != "gpu":
        raise RuntimeError(
            "PGAS smoke test requested GPU backend, but PGAS did not report backend='gpu'. "
            f"See {results_dir / 'gpu_environment.json'}."
        )

    summary_path, inference_walltime_s = _run_inference(
        repo_root=repo_root,
        data_dir=data_dir,
        results_dir=results_dir,
        env=env,
        dataset=str(args.dataset),
        trial=int(args.trial),
        methods=methods,
        plan_only=bool(args.plan_only),
    )
    result = _validate_summary(summary_path, methods, plan_only=bool(args.plan_only))
    total_walltime_s = time.perf_counter() - smoke_start
    _write_json(
        results_dir / "smoke_summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "require_gpu": bool(args.require_gpu),
            "gpu_visible": _gpu_visible(probe),
            "dataset": str(args.dataset),
            "trial": int(args.trial),
            "methods": list(methods),
            "walltime_s": total_walltime_s,
            "hardware_probe_walltime_s": probe_walltime_s,
            "inference_walltime_s": inference_walltime_s,
            "result": result,
        },
    )
    print(f"[smoke] wrote {results_dir / 'smoke_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
