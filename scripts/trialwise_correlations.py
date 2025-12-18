#!/usr/bin/env python3
"""
Compute trial-wise correlation-to-GT for method outputs saved under results/full_evaluation.

This is useful for retroactive analysis of existing runs (even if they were generated before
the built-in --trialwise-correlations flag existed).

It reads `comparison.json` to locate the exact cache entries for each method, loads the cached
`spike_prob` series, and re-computes correlations per trial window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from c_spikes.inference.eval import build_ground_truth_series, compute_trialwise_correlations
from c_spikes.inference.types import MethodResult, TrialSeries, compute_config_signature, compute_sampling_rate
from c_spikes.utils import load_Janelia_data


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", type=Path, default=Path("results/full_evaluation"), help="Root containing run_tag/dataset/smoothing.")
    p.add_argument("--data-root", type=Path, required=True, help="Root containing dataset .mat files (e.g. data/janelia_8m/excitatory).")
    p.add_argument("--run", action="append", metavar="TAG", help="Restrict to specific run tag(s). Repeatable.")
    p.add_argument("--dataset", action="append", metavar="STEM", help="Restrict to specific dataset stem(s). Repeatable.")
    p.add_argument("--smoothing", action="append", metavar="LABEL", help="Restrict to smoothing label(s) (raw, 30Hz, 10Hz). Repeatable.")
    p.add_argument("--method", action="append", metavar="NAME", help="Restrict to method label(s) (e.g., ens2, cascade). Repeatable.")
    p.add_argument("--edges-path", type=Path, default=None, help="Optional edges .npy mapping dataset_stem -> (n_trials,2).")
    p.add_argument(
        "--corr-sigma-ms",
        type=float,
        action="append",
        default=None,
        help="Gaussian sigma (ms) for correlation smoothing. Repeatable; defaults to 50 if omitted.",
    )
    p.add_argument("--out-csv", type=Path, default=Path("results/trialwise_correlations.csv"), help="Output CSV path.")
    return p.parse_args(argv)


def _uniq(values: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for v in values:
        t = str(v).strip()
        if t and t not in out:
            out.append(t)
    return out or None


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
) -> Tuple[List[Tuple[float, float]], float, float, float]:
    time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))

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
    raw_time = np.asarray(raw_time, dtype=np.float64)
    raw_time = raw_time[np.isfinite(raw_time)]
    raw_time = np.sort(raw_time)
    raw_fs = compute_sampling_rate(raw_time)

    global_start = float(min(tr.times[0] for tr in trials))
    global_end = float(max(tr.times[-1] for tr in trials))

    if edges_lookup is not None and dataset_path.stem in edges_lookup:
        edges = np.asarray(edges_lookup[dataset_path.stem], dtype=np.float64)
        windows = [(float(s), float(e)) for s, e in edges]
    else:
        windows = [(float(tr.times[0]), float(tr.times[-1])) for tr in trials]

    return windows, global_start, global_end, raw_fs


def _reference_fs_from_label(label: str, raw_fs: float) -> float:
    token = str(label).strip().lower()
    if token == "raw":
        return float(raw_fs)
    if token in {"30hz", "30"}:
        return 30.0
    if token in {"10hz", "10"}:
        return 10.0
    # fallback: attempt parse like "30.0Hz"
    if token.endswith("hz"):
        try:
            return float(token[:-2])
        except Exception:
            pass
    return float(raw_fs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    eval_root = args.eval_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    edges_lookup = _load_edges(args.edges_path)

    run_filter = _uniq(args.run)
    dataset_filter = _uniq(args.dataset)
    smoothing_filter = _uniq(args.smoothing)
    method_filter = _uniq(args.method)

    sigma_values_raw = args.corr_sigma_ms or [50.0]
    sigma_values: List[float] = []
    for sigma in sigma_values_raw:
        val = float(sigma)
        if val not in sigma_values:
            sigma_values.append(val)

    rows: List[Dict[str, Any]] = []

    if not eval_root.exists():
        raise FileNotFoundError(eval_root)

    def _looks_like_run_tag_dir(path: Path) -> bool:
        """
        Detect whether `path` is `results/full_evaluation/<run_tag>` (dataset/smoothing underneath)
        vs `results/full_evaluation` (run_tag/dataset/smoothing underneath).
        """
        for dataset_dir in path.iterdir():
            if not dataset_dir.is_dir():
                continue
            for smoothing_dir in dataset_dir.iterdir():
                if not smoothing_dir.is_dir():
                    continue
                if (smoothing_dir / "comparison.json").exists():
                    return True
        return False

    if _looks_like_run_tag_dir(eval_root):
        run_dirs = [eval_root]
    else:
        run_dirs = sorted(p for p in eval_root.iterdir() if p.is_dir())
    for run_dir in run_dirs:
        run_tag = run_dir.name
        if run_filter and run_tag not in run_filter:
            continue
        for dataset_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            dataset_stem = dataset_dir.name
            if dataset_filter and dataset_stem not in dataset_filter:
                continue
            dataset_path = data_root / f"{dataset_stem}.mat"
            if not dataset_path.exists():
                print(f"[warn] Missing dataset file: {dataset_path}")
                continue

            for smoothing_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                smoothing_label = smoothing_dir.name
                if smoothing_filter and smoothing_label not in smoothing_filter:
                    continue

                manifest_path = smoothing_dir / "comparison.json"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[warn] Failed to read {manifest_path}: {exc}")
                    continue
                method_entries = manifest.get("methods", [])
                if not isinstance(method_entries, list) or not method_entries:
                    continue

                trial_windows, global_start, global_end, raw_fs = _trial_windows_from_mat(
                    dataset_path, edges_lookup=edges_lookup
                )
                ref_fs = _reference_fs_from_label(smoothing_label, raw_fs)
                _, _, spike_times = load_Janelia_data(str(dataset_path))
                spike_times = np.asarray(spike_times, dtype=np.float64).ravel()

                method_results: List[MethodResult] = []
                label_by_name: Dict[str, str] = {}
                for entry in method_entries:
                    if not isinstance(entry, dict):
                        continue
                    label = str(entry.get("label", "")).strip()
                    method_name = str(entry.get("method", "")).strip()
                    cache_tag_raw = entry.get("cache_tag")
                    cache_tag = "" if cache_tag_raw is None else str(cache_tag_raw).strip()
                    if cache_tag.lower() == "none":
                        cache_tag = ""
                    cache_key_raw = entry.get("cache_key")
                    cache_key = str(cache_key_raw).strip() if cache_key_raw is not None else ""
                    if not cache_key:
                        cfg = entry.get("config", {})
                        if isinstance(cfg, dict) and cfg:
                            cache_key, _ = compute_config_signature(cfg)
                    sampling_rate = float(entry.get("sampling_rate", 0.0) or 0.0)
                    if not cache_tag:
                        # Some older `comparison.json` files didn't record `cache_tag` for methods
                        # (notably ENS2). Our cache layout uses dataset_stem as the cache dir.
                        cache_tag = dataset_stem
                    if not method_name or not cache_key:
                        continue
                    if method_filter and label not in method_filter and method_name not in method_filter:
                        continue

                    mat_path = Path("results/inference_cache") / method_name / cache_tag / f"{cache_key}.mat"
                    if not mat_path.exists():
                        print(f"[warn] Missing cache mat: {mat_path}")
                        continue
                    data = sio.loadmat(mat_path)
                    time_arr = np.asarray(data.get("time_stamps")).squeeze()
                    prob_arr = np.asarray(data.get("spike_prob")).squeeze()
                    if sampling_rate <= 0:
                        sampling_rate = compute_sampling_rate(np.asarray(time_arr, dtype=np.float64).ravel())
                    method_results.append(
                        MethodResult(
                            name=method_name,
                            time_stamps=time_arr,
                            spike_prob=prob_arr,
                            sampling_rate=float(sampling_rate),
                            metadata={"cache_tag": cache_tag, "cache_key": cache_key, "label": label},
                        )
                    )
                    label_by_name[method_name] = label or method_name

                if not method_results:
                    continue

                for sigma_ms in sigma_values:
                    ref_time, ref_trace = build_ground_truth_series(
                        spike_times,
                        global_start,
                        global_end,
                        reference_fs=ref_fs,
                        sigma_ms=float(sigma_ms),
                    )
                    trialwise = compute_trialwise_correlations(
                        method_results,
                        ref_time,
                        ref_trace,
                        trial_windows=trial_windows,
                        sigma_ms=float(sigma_ms),
                    )

                    for method_name, corr_list in trialwise.items():
                        label = label_by_name.get(method_name, method_name)
                        for trial_idx, ((start, end), corr) in enumerate(zip(trial_windows, corr_list)):
                            rows.append(
                                {
                                    "run": run_tag,
                                    "dataset": dataset_stem,
                                    "smoothing": smoothing_label,
                                    "method": method_name,
                                    "label": label,
                                    "corr_sigma_ms": float(sigma_ms),
                                    "trial": int(trial_idx),
                                    "start_s": float(start),
                                    "end_s": float(end),
                                    "correlation": float(corr) if np.isfinite(corr) else float("nan"),
                                }
                            )

    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run",
                "dataset",
                "smoothing",
                "method",
                "label",
                "corr_sigma_ms",
                "trial",
                "start_s",
                "end_s",
                "correlation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[trialwise] Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
