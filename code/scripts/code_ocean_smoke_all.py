#!/usr/bin/env python3
"""Short end-to-end smoke covering PGAS, ENS2, CASCADE, BiophysML aliasing, CSVs, and plotting."""

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

import numpy as np


DEFAULT_DATASET = "jGCaMP8f_ANM478349_cell04"
DEFAULT_TRIAL = 1
BASE_RUN = "code_ocean_smoke_all_base"
BIOPHYS_RUN = "code_ocean_smoke_all_biophys_ml_cache"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=os.environ.get("C_SPIKES_SMOKE_DATASET", DEFAULT_DATASET))
    parser.add_argument("--trial", type=int, default=int(os.environ.get("C_SPIKES_SMOKE_TRIAL", str(DEFAULT_TRIAL))))
    parser.add_argument("--fraction", type=float, default=float(os.environ.get("C_SPIKES_SMOKE_ALL_FRACTION", "0.1")))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_env(scratch_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(scratch_dir / "mpl_cache"))
    env.setdefault("XDG_CACHE_HOME", str(scratch_dir / "mpl_cache"))
    env.setdefault("PYTHONPYCACHEPREFIX", str(scratch_dir / "python_cache"))
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("C_SPIKES_TF_SUPPRESS_RUNTIME_STDERR", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Keep ENS2's torch path isolated from the unconditional TF preload used by older CLI runs.
    env.setdefault("C_SPIKES_TF_PRELOAD", "0")
    return env


def _run_step(
    *,
    name: str,
    cmd: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    results_dir: Path,
    plan_only: bool,
) -> dict[str, Any]:
    log_prefix = results_dir / f"smoke_all_{name}"
    payload: dict[str, Any] = {
        "name": name,
        "command": list(cmd),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    print("[smoke-all] " + " ".join(str(part) for part in cmd), flush=True)
    start = time.perf_counter()
    if plan_only:
        payload.update({"returncode": 0, "plan_only": True, "walltime_s": 0.0})
        _write_json(log_prefix.with_suffix(".json"), payload)
        return payload
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=dict(env),
        check=False,
        text=True,
        capture_output=True,
    )
    payload.update(
        {
            "returncode": int(proc.returncode),
            "signal": -int(proc.returncode) if proc.returncode < 0 else None,
            "walltime_s": time.perf_counter() - start,
            "stdout_log": str(log_prefix.with_suffix(".stdout.log")),
            "stderr_log": str(log_prefix.with_suffix(".stderr.log")),
        }
    )
    log_prefix.with_suffix(".stdout.log").write_text(proc.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(proc.stderr, encoding="utf-8")
    _write_json(log_prefix.with_suffix(".json"), payload)
    return payload


def _make_short_edges(
    *,
    data_dir: Path,
    results_dir: Path,
    dataset: str,
    trial: int,
    fraction: float,
) -> tuple[Path, float, float]:
    source = data_dir / "reference_inputs" / "edges" / "excitatory_time_stamp_edges.npy"
    edges = np.load(source, allow_pickle=True).item()
    if dataset not in edges:
        raise KeyError(f"{dataset!r} not found in {source}")
    arr = np.asarray(edges[dataset], dtype=np.float64).copy()
    if trial < 0 or trial >= int(arr.shape[0]):
        raise IndexError(f"trial {trial} out of bounds for {dataset} edges with shape {arr.shape}")
    start, end = [float(x) for x in arr[trial, :2]]
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError(f"Invalid source edge for {dataset} trial {trial}: {(start, end)}")
    short_end = start + (end - start) * float(fraction)
    arr[trial, 1] = short_end
    short_edges = dict(edges)
    short_edges[dataset] = arr
    out = results_dir / "smoke_all_edges.npy"
    np.save(out, short_edges, allow_pickle=True)
    return out, start, short_end


def _run_cli(
    *,
    repo_root: Path,
    data_dir: Path,
    results_dir: Path,
    env: Mapping[str, str],
    dataset: str,
    selection: Path,
    edges: Path,
    run_tag: str,
    methods: Sequence[str],
    ens2_root: Path,
    pgas_output_root: Path,
    name: str,
    plan_only: bool,
    eval_only: bool = False,
) -> dict[str, Any]:
    cmd: list[str] = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "c_spikes.cli.run",
        "--data-root",
        str(data_dir / "sample_data" / "janelia_8f" / "excitatory"),
        "--dataset",
        dataset,
        "--edges-path",
        str(edges),
        "--trial-selection-path",
        str(selection),
        "--cache-root",
        str(results_dir / "inference_cache"),
        "--output-root",
        str(results_dir / "full_evaluation"),
        "--pgas-output-root",
        str(pgas_output_root),
        "--pgas-constants",
        str(data_dir / "parameter_files" / "constants_GCaMP8_soma.json"),
        "--pgas-gparam",
        str(data_dir / "pgas_parameters" / "20230525_gold.dat"),
        "--pgas-bm-sigma",
        "0.03",
        "--ens2-pretrained-root",
        str(ens2_root),
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
        run_tag,
    ]
    if eval_only:
        cmd.append("--eval-only")
    for method in methods:
        cmd.extend(["--method", method])
    return _run_step(name=name, cmd=cmd, cwd=repo_root, env=env, results_dir=results_dir, plan_only=plan_only)


def _import_biophys_alias(
    *,
    repo_root: Path,
    data_dir: Path,
    results_dir: Path,
    env: Mapping[str, str],
    model_name: str,
    plan_only: bool,
) -> dict[str, Any]:
    code = r"""
import json
import os
from pathlib import Path

from c_spikes.inference.import_external import import_external_method
from c_spikes.inference.types import compute_config_signature

results_dir = Path(os.environ["C_SPIKES_SMOKE_ALL_RESULTS"])
data_root = Path(os.environ["C_SPIKES_SMOKE_ALL_DATA_ROOT"])
source_run = os.environ["C_SPIKES_SMOKE_ALL_SOURCE_RUN"]
target_run = os.environ["C_SPIKES_SMOKE_ALL_TARGET_RUN"]
model_name = os.environ["C_SPIKES_SMOKE_ALL_MODEL_NAME"]
eval_root = results_dir / "full_evaluation"
cache_root = results_dir / "inference_cache"

count = 0
for comparison_path in sorted((eval_root / source_run).glob("*/*/comparison.json")):
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    dataset = str(comparison["dataset"])
    smoothing = str(comparison["smoothing"])
    entries = [entry for entry in comparison.get("methods", []) if entry.get("method") == "ens2"]
    if len(entries) != 1:
        raise RuntimeError(f"Expected one ENS2 entry in {comparison_path}, got {len(entries)}")
    entry = entries[0]
    cache_tag = str(entry.get("cache_tag") or dataset).strip()
    config = dict(entry.get("config") or {})
    cache_key = str(entry.get("cache_key") or "").strip()
    if not cache_key:
        cache_key, _ = compute_config_signature(config)
    pred_path = cache_root / "ens2" / cache_tag / f"{cache_key}.mat"
    config["source_method"] = "ens2"
    config["source_run_tag"] = source_run
    config["source_cache_tag"] = cache_tag
    config["source_cache_key"] = cache_key
    config["method_alias"] = "biophys_ml"
    import_external_method(
        pred_path=pred_path,
        method="biophys_ml",
        dataset=dataset,
        smoothing=smoothing,
        run_tag=target_run,
        data_root=data_root,
        eval_root=eval_root,
        cache_root=cache_root,
        cache_tag=f"{model_name}/{dataset}",
        label="biophys_ml",
        config=config,
        corr_sigma_ms=50.0,
    )
    count += 1
print(f"imported {count} BiophysML alias entries")
"""
    import_env = dict(env)
    import_env.update(
        {
            "C_SPIKES_SMOKE_ALL_RESULTS": str(results_dir),
            "C_SPIKES_SMOKE_ALL_DATA_ROOT": str(data_dir / "sample_data" / "janelia_8f" / "excitatory"),
            "C_SPIKES_SMOKE_ALL_SOURCE_RUN": BIOPHYS_RUN,
            "C_SPIKES_SMOKE_ALL_TARGET_RUN": BASE_RUN,
            "C_SPIKES_SMOKE_ALL_MODEL_NAME": model_name,
        }
    )
    return _run_step(
        name="import_biophys_ml",
        cmd=[sys.executable, "-c", code],
        cwd=repo_root,
        env=import_env,
        results_dir=results_dir,
        plan_only=plan_only,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.perf_counter()
    repo_root = _repo_root()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    scratch_dir = args.scratch_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    env = _base_env(scratch_dir)
    selection = results_dir / "smoke_all_trial_selection.json"
    _write_json(selection, {str(args.dataset): [int(args.trial)]})
    short_edges, start_s, end_s = _make_short_edges(
        data_dir=data_dir,
        results_dir=results_dir,
        dataset=str(args.dataset),
        trial=int(args.trial),
        fraction=float(args.fraction),
    )
    duration_s = max(0.1, float(end_s - start_s))
    biophys_root = data_dir / "Pretrained_models" / "BiophysML" / "refbuild_biophysml_jg8f"

    steps: list[dict[str, Any]] = []
    steps.append(
        _run_cli(
            repo_root=repo_root,
            data_dir=data_dir,
            results_dir=results_dir,
            env=env,
            dataset=str(args.dataset),
            selection=selection,
            edges=short_edges,
            run_tag=BASE_RUN,
            methods=["pgas", "ens2", "cascade"],
            ens2_root=data_dir / "Pretrained_models" / "ENS2" / "ens2_published",
            pgas_output_root=results_dir / "pgas_output" / BASE_RUN,
            name="base_methods",
            plan_only=bool(args.plan_only),
        )
    )
    steps.append(
        _run_cli(
            repo_root=repo_root,
            data_dir=data_dir,
            results_dir=results_dir,
            env=env,
            dataset=str(args.dataset),
            selection=selection,
            edges=short_edges,
            run_tag=BIOPHYS_RUN,
            methods=["ens2"],
            ens2_root=biophys_root,
            pgas_output_root=results_dir / "pgas_output" / BIOPHYS_RUN,
            name="biophys_ml_cache",
            plan_only=bool(args.plan_only),
        )
    )
    steps.append(
        _import_biophys_alias(
            repo_root=repo_root,
            data_dir=data_dir,
            results_dir=results_dir,
            env=env,
            model_name=biophys_root.name,
            plan_only=bool(args.plan_only),
        )
    )
    steps.append(
        _run_cli(
            repo_root=repo_root,
            data_dir=data_dir,
            results_dir=results_dir,
            env=env,
            dataset=str(args.dataset),
            selection=selection,
            edges=short_edges,
            run_tag=BASE_RUN,
            methods=["pgas", "ens2", "cascade", "biophys_ml"],
            ens2_root=data_dir / "Pretrained_models" / "ENS2" / "ens2_published",
            pgas_output_root=results_dir / "pgas_output" / BASE_RUN,
            name="eval_with_biophys_ml",
            plan_only=bool(args.plan_only),
            eval_only=True,
        )
    )

    trialwise_csv = results_dir / "trialwise_correlations_smoke_all.csv"
    steps.append(
        _run_step(
            name="trialwise_csv",
            cmd=[
                sys.executable,
                "code/scripts/trialwise_correlations.py",
                "--eval-root",
                str(results_dir / "full_evaluation"),
                "--cache-root",
                str(results_dir / "inference_cache"),
                "--data-root",
                str(data_dir / "sample_data" / "janelia_8f" / "excitatory"),
                "--edges-path",
                str(short_edges),
                "--out-csv",
                str(trialwise_csv),
                "--corr-sigma-ms",
                "50",
                "--run",
                BASE_RUN,
                "--dataset",
                str(args.dataset),
            ],
            cwd=repo_root,
            env=env,
            results_dir=results_dir,
            plan_only=bool(args.plan_only),
        )
    )
    plot_path = results_dir / "plots" / "smoke_all_trace_panel.png"
    steps.append(
        _run_step(
            name="plot_trace_panel",
            cmd=[
                sys.executable,
                "code/scripts/plot_trialwise_trace_panel.py",
                "--csv",
                str(trialwise_csv),
                "--eval-root",
                str(results_dir / "full_evaluation"),
                "--data-root",
                str(data_dir / "sample_data" / "janelia_8f" / "excitatory"),
                "--edges-path",
                str(short_edges),
                "--dataset",
                str(args.dataset),
                "--out",
                str(plot_path),
                "--method",
                f"pgas@{BASE_RUN}",
                "--method",
                f"biophys_ml@{BASE_RUN}",
                "--method",
                f"ens2=ens2@{BASE_RUN}",
                "--method",
                f"cascade@{BASE_RUN}",
                "--smoothing",
                "raw",
                "--corr-sigma-ms",
                "50",
                "--trial",
                str(int(args.trial)),
                "--start-s",
                str(float(start_s)),
                "--duration-s",
                str(float(duration_s)),
                "--title",
                "smoke_all trace panel",
            ],
            cwd=repo_root,
            env=env,
            results_dir=results_dir,
            plan_only=bool(args.plan_only),
        )
    )

    failures = [step for step in steps if int(step.get("returncode", 1)) != 0]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "trial": int(args.trial),
        "fraction": float(args.fraction),
        "short_edge": [float(start_s), float(end_s)],
        "methods": ["pgas", "biophys_ml", "ens2", "cascade"],
        "trialwise_csv": str(trialwise_csv),
        "plot": str(plot_path),
        "walltime_s": time.perf_counter() - start,
        "steps": steps,
    }
    _write_json(results_dir / "smoke_all_summary.json", summary)
    print(f"[smoke-all] wrote {results_dir / 'smoke_all_summary.json'}", flush=True)
    if failures:
        detail = "; ".join(f"{step['name']} rc={step['returncode']} stderr={step.get('stderr_log')}" for step in failures)
        raise RuntimeError(f"smoke-all failures: {detail}")


if __name__ == "__main__":
    main()
