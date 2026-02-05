#!/usr/bin/env python3
"""Compare PGAS CPU vs GPU outputs on a single data snippet."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from c_spikes.inference.pgas import (
    PGAS_BM_SIGMA_DEFAULT,
    PGAS_BURNIN,
    PGAS_NITER,
    PgasConfig,
    run_pgas_inference,
)
from c_spikes.inference.types import TrialSeries
from c_spikes.utils import load_Janelia_data, unroll_mean_pgas_traj


@dataclass
class RunSpec:
    name: str
    module: str


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"none", "null", "auto", "estimate", "estimated"}:
        return None
    return float(value)


def _select_snippet(
    time_stamps: np.ndarray,
    dff: np.ndarray,
    *,
    trial_index: int,
    start_s: float | None,
    duration_s: float | None,
    end_s: float | None,
) -> tuple[TrialSeries, tuple[float, float]]:
    if trial_index < 0 or trial_index >= time_stamps.shape[0]:
        raise ValueError(f"trial_index {trial_index} out of range for {time_stamps.shape[0]} trials.")

    times = np.asarray(time_stamps[trial_index], dtype=float)
    values = np.asarray(dff[trial_index], dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if times.size < 2:
        raise ValueError(f"Trial {trial_index} has insufficient samples after NaN filtering.")

    t_min = float(times[0])
    t_max = float(times[-1])

    if start_s is None:
        start_s = t_min
    if start_s < t_min:
        print(f"[WARN] snippet start {start_s:.3f}s < trial start {t_min:.3f}s; clamping.")
        start_s = t_min

    if end_s is None:
        if duration_s is None:
            end_s = t_max
        else:
            end_s = start_s + float(duration_s)
    if end_s > t_max:
        print(f"[WARN] snippet end {end_s:.3f}s > trial end {t_max:.3f}s; clamping.")
        end_s = t_max

    if end_s <= start_s:
        raise ValueError(f"Invalid snippet window: start={start_s}, end={end_s}.")

    win_mask = (times >= start_s) & (times <= end_s)
    if np.count_nonzero(win_mask) < 2:
        raise ValueError(
            f"Snippet window [{start_s:.3f}, {end_s:.3f}] has too few samples in trial {trial_index}."
        )

    trial = TrialSeries(times=times[win_mask].copy(), values=values[win_mask].copy())
    return trial, (float(start_s), float(end_s))


def _extract_spike_times(ap_times: np.ndarray, trial_index: int) -> np.ndarray:
    arr = np.asarray(ap_times)
    if arr.dtype == object:
        arr = arr.squeeze()
        if arr.ndim == 0:
            spikes = np.asarray(arr.item(), dtype=float).ravel()
        else:
            if trial_index >= arr.size:
                raise ValueError(f"trial_index {trial_index} out of range for ap_times shape {arr.shape}.")
            spikes = np.asarray(arr[trial_index], dtype=float).ravel()
    else:
        spikes = arr.astype(float).ravel()
    return spikes


def _find_latest(pattern: str, root: Path) -> Path:
    matches = list(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No matches for {pattern} in {root}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _load_param_samples(path: Path) -> Tuple[List[str], np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"No samples in {path}")
    if data.ndim == 0:
        data = np.array([tuple(data.tolist())], dtype=data.dtype)
    fields = list(data.dtype.names or [])
    values = np.column_stack([data[name] for name in fields])
    return fields, values


def _compare_param_distributions(
    fields: List[str],
    cpu_vals: np.ndarray,
    gpu_vals: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    from scipy.stats import ks_2samp

    out: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(fields):
        cpu = cpu_vals[:, idx]
        gpu = gpu_vals[:, idx]
        stat, pval = ks_2samp(cpu, gpu, alternative="two-sided", mode="auto")
        out[name] = {
            "cpu_mean": float(np.mean(cpu)),
            "gpu_mean": float(np.mean(gpu)),
            "cpu_std": float(np.std(cpu)),
            "gpu_std": float(np.std(gpu)),
            "mean_diff": float(np.mean(cpu) - np.mean(gpu)),
            "ks_stat": float(stat),
            "ks_pvalue": float(pval),
        }
    return out


def _compare_spike_traces(cpu_spikes: np.ndarray, gpu_spikes: np.ndarray) -> Dict[str, float]:
    n = min(cpu_spikes.size, gpu_spikes.size)
    if n == 0:
        raise ValueError("Empty spike traces for comparison.")
    cpu = cpu_spikes[:n]
    gpu = gpu_spikes[:n]
    corr = float(np.corrcoef(cpu, gpu)[0, 1]) if n > 1 else float("nan")
    mae = float(np.mean(np.abs(cpu - gpu)))
    rmse = float(np.sqrt(np.mean((cpu - gpu) ** 2)))
    return {"corr": corr, "mae": mae, "rmse": rmse}


def _compare_map_spikes(cpu_map: np.ndarray, gpu_map: np.ndarray) -> Dict[str, float]:
    n = min(cpu_map.size, gpu_map.size)
    if n == 0:
        raise ValueError("Empty MAP spike traces for comparison.")
    cpu = cpu_map[:n]
    gpu = gpu_map[:n]
    cpu_bin = cpu > 0
    gpu_bin = gpu > 0
    intersection = float(np.sum(cpu_bin & gpu_bin))
    union = float(np.sum(cpu_bin | gpu_bin))
    jaccard = intersection / union if union > 0 else 1.0
    same_bin = float(np.mean(cpu_bin == gpu_bin))
    exact_match = float(np.mean(cpu == gpu))
    return {
        "jaccard": jaccard,
        "same_bin_fraction": same_bin,
        "exact_match_fraction": exact_match,
    }


def _run_worker(spec: RunSpec, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend-name",
        spec.name,
        "--backend-module",
        spec.module,
        "--dataset",
        str(args.dataset),
        "--trial-index",
        str(args.trial_index),
        "--snippet-duration-s",
        str(args.snippet_duration_s),
        "--pgas-constants",
        str(args.pgas_constants),
        "--pgas-gparam",
        str(args.pgas_gparam),
        "--pgas-output-root",
        str(args.pgas_output_root),
        "--pgas-bm-sigma",
        str(args.pgas_bm_sigma),
        "--niter",
        str(args.niter),
        "--bm-sigma-gap-s",
        str(args.bm_sigma_gap_s),
        "--run-name",
        str(args.run_name),
    ]
    if args.snippet_start_s is not None:
        cmd.extend(["--snippet-start-s", str(args.snippet_start_s)])
    if args.snippet_end_s is not None:
        cmd.extend(["--snippet-end-s", str(args.snippet_end_s)])
    if args.pgas_resample is not None:
        cmd.extend(["--pgas-resample", str(args.pgas_resample)])
    if args.use_cache:
        cmd.append("--use-cache")

    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to .mat dataset.")
    parser.add_argument("--trial-index", type=int, default=0, help="Trial/epoch index (0-based).")
    parser.add_argument("--snippet-start-s", type=float, default=None, help="Snippet start time (sec).")
    parser.add_argument(
        "--snippet-duration-s",
        type=float,
        default=10.0,
        help="Snippet duration (sec). Default: 10s.",
    )
    parser.add_argument("--snippet-end-s", type=float, default=None, help="Snippet end time (sec).")
    parser.add_argument(
        "--backend-cpu",
        type=str,
        default="c_spikes.pgas.pgas_bound_cpu",
        help="CPU backend module path.",
    )
    parser.add_argument(
        "--backend-gpu",
        type=str,
        default="c_spikes.pgas.pgas_bound_gpu",
        help="GPU backend module path.",
    )
    parser.add_argument(
        "--pgas-constants",
        type=Path,
        default=Path("parameter_files/constants_GCaMP8_soma.json"),
        help="PGAS constants JSON.",
    )
    parser.add_argument(
        "--pgas-gparam",
        type=Path,
        default=Path("src/c_spikes/pgas/20230525_gold.dat"),
        help="PGAS GCaMP parameter file.",
    )
    parser.add_argument(
        "--pgas-output-root",
        type=Path,
        default=Path("results/pgas_cpu_gpu_compare"),
        help="Output root for PGAS run files + summaries.",
    )
    parser.add_argument(
        "--pgas-bm-sigma",
        type=str,
        default=str(PGAS_BM_SIGMA_DEFAULT),
        help="Fixed bm_sigma value, or 'auto' to estimate.",
    )
    parser.add_argument("--pgas-resample", type=float, default=None, help="PGAS resample Hz (None=raw).")
    parser.add_argument("--niter", type=int, default=PGAS_NITER, help="PGAS niter.")
    parser.add_argument(
        "--bm-sigma-gap-s",
        type=float,
        default=0.15,
        help="Gap (s) around spikes excluded when estimating bm_sigma.",
    )
    parser.add_argument("--use-cache", action="store_true", help="Allow cache loads (default: off).")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name (default: timestamp-based).",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--backend-module", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_main(args)
        return

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    args.run_name = run_name
    run_root = args.pgas_output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    cpu_spec = RunSpec("cpu", args.backend_cpu)
    gpu_spec = RunSpec("gpu", args.backend_gpu)

    _run_worker(cpu_spec, args)
    _run_worker(gpu_spec, args)

    # Load outputs and compare
    cpu_root = run_root / cpu_spec.name
    gpu_root = run_root / gpu_spec.name

    param_cpu = _find_latest(f"param_samples_*_{cpu_spec.name}_*trial0.dat", cpu_root)
    param_gpu = _find_latest(f"param_samples_*_{gpu_spec.name}_*trial0.dat", gpu_root)

    fields_cpu, vals_cpu = _load_param_samples(param_cpu)
    fields_gpu, vals_gpu = _load_param_samples(param_gpu)
    if fields_cpu != fields_gpu:
        raise ValueError("CPU/GPU param sample headers differ; cannot compare distributions.")

    traj_cpu = _find_latest(f"traj_samples_*_{cpu_spec.name}_*trial0.dat", cpu_root)
    logp_cpu = _find_latest(f"logp_*_{cpu_spec.name}_*trial0.dat", cpu_root)
    traj_gpu = _find_latest(f"traj_samples_*_{gpu_spec.name}_*trial0.dat", gpu_root)
    logp_gpu = _find_latest(f"logp_*_{gpu_spec.name}_*trial0.dat", gpu_root)

    _, _, spikes_mean_cpu, _, spikes_map_cpu = unroll_mean_pgas_traj(
        str(traj_cpu), str(logp_cpu), burnin=PGAS_BURNIN
    )
    _, _, spikes_mean_gpu, _, spikes_map_gpu = unroll_mean_pgas_traj(
        str(traj_gpu), str(logp_gpu), burnin=PGAS_BURNIN
    )

    param_summary = _compare_param_distributions(fields_cpu, vals_cpu, vals_gpu)
    mean_trace_metrics = _compare_spike_traces(
        np.asarray(spikes_mean_cpu, dtype=float),
        np.asarray(spikes_mean_gpu, dtype=float),
    )
    map_metrics = _compare_map_spikes(
        np.asarray(spikes_map_cpu, dtype=float),
        np.asarray(spikes_map_gpu, dtype=float),
    )

    summary = {
        "dataset": str(args.dataset),
        "trial_index": int(args.trial_index),
        "snippet_duration_s": float(args.snippet_duration_s),
        "pgas_constants": str(args.pgas_constants),
        "pgas_gparam": str(args.pgas_gparam),
        "pgas_resample": args.pgas_resample,
        "niter": int(args.niter),
        "burnin": int(PGAS_BURNIN),
        "param_samples_cpu": str(param_cpu),
        "param_samples_gpu": str(param_gpu),
        "traj_cpu": str(traj_cpu),
        "traj_gpu": str(traj_gpu),
        "logp_cpu": str(logp_cpu),
        "logp_gpu": str(logp_gpu),
        "parameter_distributions": param_summary,
        "mean_spike_trace": mean_trace_metrics,
        "map_spikes": map_metrics,
    }

    out_path = run_root / "compare_summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("[DONE] Wrote summary to", out_path)
    print("Mean spike trace corr:", mean_trace_metrics["corr"])
    print("MAP spike Jaccard:", map_metrics["jaccard"])


def _worker_main(args: argparse.Namespace) -> None:
    if args.backend_name is None or args.backend_module is None:
        raise ValueError("Worker mode requires --backend-name and --backend-module.")
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    time_stamps, dff, ap_times = load_Janelia_data(str(args.dataset))
    trial, (start_s, end_s) = _select_snippet(
        time_stamps,
        dff,
        trial_index=args.trial_index,
        start_s=args.snippet_start_s,
        duration_s=args.snippet_duration_s,
        end_s=args.snippet_end_s,
    )
    spike_times = _extract_spike_times(ap_times, args.trial_index)
    spike_times = spike_times[(spike_times >= start_s) & (spike_times <= end_s)]

    raw_fs = trial.current_fs()
    dataset_stem = args.dataset.stem
    snippet_label = f"snip{end_s - start_s:.2f}s"
    dataset_tag = f"{dataset_stem}_{args.backend_name}"

    output_root = args.pgas_output_root / args.run_name / args.backend_name
    output_root.mkdir(parents=True, exist_ok=True)

    bm_sigma = _parse_optional_float(args.pgas_bm_sigma)

    pgas_cfg = PgasConfig(
        dataset_tag=dataset_tag,
        output_root=output_root,
        constants_file=args.pgas_constants,
        gparam_file=args.pgas_gparam,
        resample_fs=args.pgas_resample,
        niter=args.niter,
        burnin=PGAS_BURNIN,
        downsample_label=snippet_label,
        maxspikes=None,
        bm_sigma=bm_sigma,
        bm_sigma_gap_s=args.bm_sigma_gap_s,
        edges=None,
        use_cache=args.use_cache,
    )

    # Patch backend module into the import path expected by run_pgas_inference.
    import importlib
    import sys as _sys

    backend_module = importlib.import_module(args.backend_module)
    _sys.modules["c_spikes.pgas.pgas_bound"] = backend_module

    _ = run_pgas_inference(
        trials=[trial],
        raw_fs=raw_fs,
        spike_times=spike_times,
        config=pgas_cfg,
    )


if __name__ == "__main__":
    main()
