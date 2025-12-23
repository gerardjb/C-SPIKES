#!/usr/bin/env python3
"""
Stacked trace panel for one representative trial window.

This script is meant to recreate a "paper figure" style panel like the provided example:
  - Top: fluorescence (ΔF/F) for a selected snippet of one trial
  - Next: ground truth spike train + dotted vertical lines at spike times
  - Below: spike_prob traces for multiple methods, each normalized to [0, 1] within the snippet
           and vertically offset so traces don't overlap.
  - Right of each method trace: Pearson r for that snippet (vs smoothed GT).

Trial selection:
  - Uses `results/trialwise_correlations.csv` (from scripts/trialwise_correlations.py)
  - Picks the trial index whose (method-by-method) trialwise correlations are collectively
    closest to each method's median (minimizes sum |corr - median| across methods).

Because methods may live under different full-evaluation run tags (e.g. pgasraw vs cascadein_ens2),
you can map each method to a run tag via --run-by-method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# In some sandboxed/HPC environments Intel OpenMP shared-memory init can fail.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ.setdefault("MPLBACKEND", "Agg")
_default_cache_root = Path.cwd() / "tmp" / "mpl_cache"
try:
    _default_cache_root.mkdir(parents=True, exist_ok=True)
except OSError:
    _default_cache_root = Path("/tmp") / "c_spikes_mpl_cache"
    _default_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_default_cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(_default_cache_root))

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

from c_spikes.inference.eval import build_ground_truth_series, resample_prediction_to_reference
from c_spikes.inference.types import MethodResult, TrialSeries, compute_config_signature, compute_sampling_rate
from c_spikes.model_eval.model_eval import smooth_prediction
from c_spikes.utils import load_Janelia_data


DEFAULT_COLORS = {
    "pgas": "#009E73",
    "cascade": "#F3AE14",
    "ens2": "#A6780C",
}

DEFAULT_LABELS = {
    "pgas": r"Biophys$_{SMC}$",
    "cascade": "CASCADE",
    "ens2": r"ENS$^2$",
    # Future-proofing:
    "mlspike": "MLspike",
    "biophys_ml": r"Biophys$_{ML}$",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("results/trialwise_correlations.csv"), help="Trialwise CSV path.")
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path("results/full_evaluation"),
        help="Full evaluation root containing run_tag/dataset/smoothing/comparison.json.",
    )
    p.add_argument("--data-root", type=Path, required=True, help="Dataset root containing .mat files.")
    p.add_argument("--edges-path", type=Path, default=None, help="Optional edges .npy mapping dataset->(n_trials,2).")
    p.add_argument("--dataset", required=True, help="Dataset stem (e.g. jGCaMP8m_ANM472179_cell02).")
    p.add_argument("--smoothing", default="raw", help="Smoothing label (raw/30Hz/10Hz).")
    p.add_argument("--corr-sigma-ms", type=float, default=50.0, help="Correlation smoothing sigma (ms).")
    p.add_argument(
        "--method",
        action="append",
        default=None,
        help="Method(s) to plot (repeatable). Default: pgas, cascade, ens2",
    )
    p.add_argument(
        "--run",
        default=None,
        help="Default run tag for all methods (if --run-by-method not provided).",
    )
    p.add_argument(
        "--run-by-method",
        action="append",
        default=None,
        metavar="METHOD=RUN_TAG",
        help="Map a specific method to a specific run tag. Repeatable.",
    )
    p.add_argument("--trial", type=int, default=None, help="Force a trial index instead of auto-selecting.")
    p.add_argument("--duration-s", type=float, default=5.0, help="Snippet duration in seconds.")
    p.add_argument(
        "--start-s",
        type=float,
        default=None,
        help="Optional snippet start time (seconds); overrides centering behavior.",
    )
    p.add_argument(
        "--end-s",
        type=float,
        default=None,
        help="Optional snippet end time (seconds); requires --start-s and overrides --duration-s.",
    )
    p.add_argument(
        "--center",
        choices=["median_spike", "trial_mid"],
        default="median_spike",
        help="How to center the snippet inside the chosen trial window.",
    )
    p.add_argument(
        "--display-sigma-ms",
        type=float,
        default=None,
        help="Gaussian sigma (ms) for display traces. Defaults to --corr-sigma-ms.",
    )
    p.add_argument(
        "--gt-sigma-ms",
        type=float,
        default=None,
        help="Alias for --display-sigma-ms (kept for backwards compatibility).",
    )
    p.add_argument(
        "--scalebar-time-s",
        type=float,
        default=5.0,
        help="Time scalebar length (s). Default matches the 5s window.",
    )
    p.add_argument(
        "--method-label-x-offset-frac",
        type=float,
        default=0.0,
        help="Horizontal offset for left labels as a fraction of the snippet duration (default: 0.0).",
    )
    p.add_argument("--scalebar-dff", type=float, default=0.5, help="ΔF/F scalebar size (used for display scaling).")
    p.add_argument("--title", type=str, default="Excitatory cell sample", help="Figure title.")
    p.add_argument("--out", type=Path, default=Path("results/trialwise_trace_panel.png"), help="Output PNG path.")
    p.add_argument("--figsize", type=float, nargs=2, default=(3.4, 5.2), help="Figure size in inches.")
    p.add_argument("--dpi", type=int, default=250, help="Output DPI.")
    return p.parse_args(argv)


def _load_edges(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    edges_path = path.expanduser().resolve()
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)
    return np.load(edges_path, allow_pickle=True).item()


def _trial_windows_from_mat(
    dataset_path: Path,
    *,
    edges_lookup: Optional[Dict[str, Any]],
) -> Tuple[List[Tuple[float, float]], float]:
    time_stamps, dff, _spike_times = load_Janelia_data(str(dataset_path))

    trials: List[TrialSeries] = []
    for idx in range(time_stamps.shape[0]):
        t = np.asarray(time_stamps[idx], dtype=np.float64)
        y = np.asarray(dff[idx], dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        if t.size == 0:
            continue
        trials.append(TrialSeries(times=t, values=y))
    if not trials:
        raise RuntimeError(f"No valid trials for dataset {dataset_path.stem}")

    raw_time = np.concatenate([tr.times for tr in trials])
    raw_fs = float(compute_sampling_rate(raw_time))

    if edges_lookup is not None and dataset_path.stem in edges_lookup:
        edges = np.asarray(edges_lookup[dataset_path.stem], dtype=np.float64)
        windows = [(float(s), float(e)) for s, e in edges]
    else:
        windows = [(float(tr.times[0]), float(tr.times[-1])) for tr in trials]

    return windows, raw_fs


def _reference_fs_from_label(label: str, raw_fs: float) -> float:
    token = str(label).strip().lower()
    if token == "raw":
        return float(raw_fs)
    if token in {"30hz", "30"}:
        return 30.0
    if token in {"10hz", "10"}:
        return 10.0
    if token.endswith("hz"):
        try:
            return float(token[:-2])
        except Exception:
            pass
    return float(raw_fs)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def _float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def _parse_run_by_method(items: Optional[Sequence[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not items:
        return mapping
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected METHOD=RUN_TAG, got: {item!r}")
        method, run = item.split("=", 1)
        method = method.strip()
        run = run.strip()
        if not method or not run:
            raise ValueError(f"Expected METHOD=RUN_TAG, got: {item!r}")
        mapping[method] = run
    return mapping


@dataclass(frozen=True)
class CacheSpec:
    method: str
    run_tag: str
    dataset: str
    smoothing: str


def _load_comparison_method_entry(
    eval_root: Path,
    spec: CacheSpec,
) -> Dict[str, Any]:
    cmp_path = eval_root / spec.run_tag / spec.dataset / spec.smoothing / "comparison.json"
    if not cmp_path.exists():
        raise FileNotFoundError(f"Missing comparison.json: {cmp_path}")
    obj = json.loads(cmp_path.read_text(encoding="utf-8"))
    for entry in obj.get("methods", []):
        if isinstance(entry, dict) and str(entry.get("method", "")).strip() == spec.method:
            return entry
    raise KeyError(f"Method {spec.method!r} not found in {cmp_path}")


def _cache_paths_from_entry(
    entry: Dict[str, Any],
    *,
    dataset_fallback: str,
) -> Tuple[str, str]:
    method_name = str(entry.get("method", "")).strip()
    cache_tag_raw = entry.get("cache_tag")
    cache_tag = "" if cache_tag_raw is None else str(cache_tag_raw).strip()
    if cache_tag.lower() == "none":
        cache_tag = ""
    if not cache_tag:
        cache_tag = dataset_fallback

    cache_key_raw = entry.get("cache_key")
    cache_key = str(cache_key_raw).strip() if cache_key_raw is not None else ""
    if not cache_key:
        cfg = entry.get("config", {})
        if isinstance(cfg, dict) and cfg:
            cache_key, _ = compute_config_signature(cfg)
    if not method_name or not cache_key or not cache_tag:
        raise ValueError("Could not resolve method cache path from comparison entry.")
    return cache_tag, cache_key


def _load_method_cache_mat(method: str, cache_tag: str, cache_key: str) -> MethodResult:
    mat_path = Path("results/inference_cache") / method / cache_tag / f"{cache_key}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing cache mat: {mat_path}")
    data = sio.loadmat(mat_path)
    time_arr = np.asarray(data.get("time_stamps")).squeeze()
    prob_arr = np.asarray(data.get("spike_prob")).squeeze()
    fs = float(compute_sampling_rate(np.asarray(time_arr, dtype=np.float64).ravel()))
    return MethodResult(name=method, time_stamps=time_arr, spike_prob=prob_arr, sampling_rate=fs, metadata={})


def _segment_slices(times: np.ndarray, fs_est: float, gap_factor: float = 4.0) -> List[slice]:
    times = np.asarray(times, dtype=np.float64).ravel()
    if times.size == 0:
        return []
    diffs = np.diff(times)
    base_dt = 1.0 / float(fs_est)
    threshold = float(gap_factor) * base_dt
    breaks = np.where((diffs > threshold) | ~np.isfinite(diffs))[0] + 1
    idx = np.concatenate([[0], breaks, [times.size]])
    segs: List[slice] = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a:
            segs.append(slice(int(a), int(b)))
    return segs


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.zeros_like(v)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(v)
    out = (v - lo) / (hi - lo)
    out[~np.isfinite(out)] = np.nan
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from c_spikes.viz.trialwise_plots import plot_trace_panel

    display_sigma_ms = None
    if args.display_sigma_ms is not None:
        display_sigma_ms = float(args.display_sigma_ms)
    if args.gt_sigma_ms is not None:
        display_sigma_ms = float(args.gt_sigma_ms)

    fig, _ax, meta = plot_trace_panel(
        csv_path=args.csv,
        eval_root=args.eval_root,
        data_root=args.data_root,
        edges_path=args.edges_path,
        dataset=args.dataset,
        smoothing=args.smoothing,
        corr_sigma_ms=float(args.corr_sigma_ms),
        display_sigma_ms=display_sigma_ms,
        methods=args.method,
        run=args.run,
        run_by_method=args.run_by_method,
        trial=args.trial,
        duration_s=float(args.duration_s),
        start_s=args.start_s,
        end_s=args.end_s,
        center=str(args.center),
        method_label_x_offset_frac=float(args.method_label_x_offset_frac),
        scalebar_time_s=float(args.scalebar_time_s),
        scalebar_dff=float(args.scalebar_dff),
        title=str(args.title),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
    )
    fig.savefig(out_path)
    print(f"[plot] Wrote {out_path}")
    print(
        f"[plot] Selected trial={meta['trial']} window=({meta['window_start_s']:.3f},{meta['window_end_s']:.3f}) "
        f"run_by_method={meta['run_by_method']}"
    )


if __name__ == "__main__":
    main()
