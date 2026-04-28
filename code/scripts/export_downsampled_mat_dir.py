#!/usr/bin/env python3
"""
Export downsampled Janelia-style ground-truth datasets for external tools (e.g., MLspike).

This reads a directory of `.mat` files containing (at minimum):
  - time_stamps: (n_trials, n_samples) seconds
  - dff:         (n_trials, n_samples)
  - ap_times:    (n_spikes,) or (1, n_spikes) seconds (optional but recommended)

and writes a new directory tree containing downsampled copies at requested sampling rates.

Downsampling semantics match the repo's inference smoothing:
  - if fs/target_fs is ~integer, mean-pool contiguous blocks
  - otherwise, linearly interpolate onto an evenly spaced time grid

Example (export 30Hz + 10Hz under results/gt_downsampled/):
  PYTHONPATH=src python scripts/export_downsampled_mat_dir.py \\
    --data-root data/janelia_8f/excitatory \\
    --out-root results/gt_downsampled/janelia_8f_excitatory \\
    --smoothing-level 30Hz --smoothing-level 10Hz

Optional: restrict each trial to precomputed per-trial windows (`edges` dict .npy):
  PYTHONPATH=src python scripts/export_downsampled_mat_dir.py \\
    --data-root data/janelia_8f/excitatory \\
    --out-root results/gt_downsampled/janelia_8f_excitatory_windowed \\
    --smoothing-level 30Hz \\
    --edges-path results/excitatory_time_stamp_edges.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio


def _format_fs_label(target_fs: float) -> str:
    fs = float(target_fs)
    if np.isclose(fs, round(fs)):
        return f"{int(round(fs))}Hz"
    token = f"{fs:.2f}".rstrip("0").rstrip(".")
    return token.replace(".", "p") + "Hz"


def _resolve_levels(
    smoothing_levels: Optional[Sequence[str]],
    target_fs: Optional[Sequence[float]],
) -> List[Tuple[str, Optional[float]]]:
    levels: List[Tuple[str, Optional[float]]] = []
    if smoothing_levels:
        from c_spikes.inference.smoothing import resolve_smoothing_levels

        levels.extend(list(resolve_smoothing_levels(list(smoothing_levels))))
    if target_fs:
        for fs in target_fs:
            fs = float(fs)
            if fs <= 0:
                raise ValueError("--target-fs must be positive.")
            levels.append((_format_fs_label(fs), fs))
    if not levels:
        raise ValueError("Provide at least one --smoothing-level or --target-fs.")
    # Deduplicate by label (first wins).
    seen: set[str] = set()
    unique: List[Tuple[str, Optional[float]]] = []
    for label, fs in levels:
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, fs))
    return unique


def _iter_mat_files(root: Path, globs: Sequence[str]) -> Iterable[Path]:
    paths: List[Path] = []
    for pattern in globs:
        paths.extend(sorted(root.glob(pattern)))
    return sorted(set(paths))


def _load_edges_dict(edges_path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(edges_path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a .npy dict, got .npz: {edges_path}")
    try:
        edges_dict = payload.item()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to load edges dict from {edges_path}") from exc
    if not isinstance(edges_dict, dict):
        raise ValueError(f"Edges file did not contain a dict: {edges_path}")
    return {str(k): np.asarray(v) for k, v in edges_dict.items()}


def _trim_by_edges(times: np.ndarray, values: np.ndarray, start: float, end: float, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError(f"Non-finite edges ({start}, {end}).")
    if end <= start:
        raise ValueError(f"Edge end must exceed start: ({start}, {end}).")
    mask = (times >= start - tol) & (times <= end + tol)
    if not mask.any():
        raise ValueError(f"No samples within edges window ({start}, {end}).")
    return times[mask], values[mask]


def _pad_trials(
    trials: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    pad_mode: str,
    fallback_dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not trials:
        raise ValueError("No trials to pad.")
    lengths = np.asarray([t.size for t, _ in trials], dtype=np.int64)
    max_len = int(lengths.max(initial=0))
    if max_len <= 0:
        raise ValueError("All trials are empty.")

    if pad_mode not in {"nan", "edge"}:
        raise ValueError("--pad-mode must be one of: nan, edge")

    if pad_mode == "nan":
        time_out = np.full((len(trials), max_len), np.nan, dtype=np.float64)
        trace_out = np.full((len(trials), max_len), np.nan, dtype=np.float32)
        for idx, (times, values) in enumerate(trials):
            n = int(times.size)
            if n == 0:
                continue
            time_out[idx, :n] = np.asarray(times, dtype=np.float64)
            trace_out[idx, :n] = np.asarray(values, dtype=np.float32)
        return time_out, trace_out, lengths

    time_out = np.zeros((len(trials), max_len), dtype=np.float64)
    trace_out = np.zeros((len(trials), max_len), dtype=np.float32)
    for idx, (times, values) in enumerate(trials):
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if times.size == 0:
            continue
        if times.size < max_len:
            if times.size > 1:
                dt = float(np.median(np.diff(times)))
            else:
                dt = float(fallback_dt or 0.0)
            if not np.isfinite(dt) or dt <= 0:
                dt = float(fallback_dt or 0.0)
            if dt <= 0:
                raise ValueError("Unable to infer pad dt; provide non-empty trial or use --pad-mode nan.")
            pad = max_len - int(times.size)
            extra_times = times[-1] + np.arange(1, pad + 1, dtype=np.float64) * dt
            extra_values = np.full(pad, values[-1], dtype=np.float64)
            times = np.concatenate([times, extra_times])
            values = np.concatenate([values, extra_values])
        time_out[idx] = times
        trace_out[idx] = values.astype(np.float32, copy=False)
    return time_out, trace_out, lengths


def export_one(
    mat_path: Path,
    *,
    out_path: Path,
    target_fs: Optional[float],
    edges: Optional[np.ndarray],
    pad_mode: str,
    overwrite: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return

    data = sio.loadmat(mat_path)
    if "time_stamps" not in data or "dff" not in data:
        raise KeyError(f"Missing required keys in {mat_path} (need time_stamps + dff).")
    time_stamps = np.asarray(data["time_stamps"], dtype=np.float64)
    dff = np.asarray(data["dff"], dtype=np.float64)
    if time_stamps.ndim != 2 or dff.ndim != 2 or time_stamps.shape != dff.shape:
        raise ValueError(f"Unexpected shapes in {mat_path}: time_stamps={time_stamps.shape} dff={dff.shape}")

    spike_times = np.asarray(data.get("ap_times", np.asarray([], dtype=np.float64)), dtype=np.float64).ravel()

    trials: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(time_stamps.shape[0]):
        t = np.asarray(time_stamps[idx], dtype=np.float64)
        y = np.asarray(dff[idx], dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        if t.size == 0:
            raise ValueError(f"Empty/invalid trial {idx} in {mat_path}")
        if edges is not None:
            start, end = map(float, edges[idx])
            t, y = _trim_by_edges(t, y, start, end)
        if target_fs is not None:
            from c_spikes.inference.smoothing import mean_downsample_trace

            ds = mean_downsample_trace(t, y, float(target_fs))
            t = np.asarray(ds.times, dtype=np.float64)
            y = np.asarray(ds.values, dtype=np.float64)
        trials.append((t, y))

    fallback_dt = None
    if target_fs is not None and float(target_fs) > 0:
        fallback_dt = 1.0 / float(target_fs)
    times_out, dff_out, lengths = _pad_trials(trials, pad_mode=pad_mode, fallback_dt=fallback_dt)

    if edges is not None and spike_times.size:
        mask = np.zeros(spike_times.shape, dtype=bool)
        for start, end in np.asarray(edges, dtype=float):
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            mask |= (spike_times >= float(start)) & (spike_times <= float(end))
        spike_times = spike_times[mask]

    payload = {
        "time_stamps": times_out,
        "dff": dff_out,
        "ap_times": spike_times.reshape(1, -1),
        "valid_lengths": lengths.reshape(1, -1),
    }
    if edges is not None:
        payload["edges"] = np.asarray(edges, dtype=np.float64)
    if target_fs is not None:
        payload["target_fs_hz"] = float(target_fs)
    sio.savemat(out_path, payload, do_compression=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="Directory containing input *.mat datasets.")
    p.add_argument("--out-root", type=Path, required=True, help="Output root directory (subdirs per smoothing).")
    p.add_argument(
        "--dataset-glob",
        action="append",
        default=None,
        help="Glob(s) under --data-root selecting datasets to export (repeatable). Default: '*.mat'.",
    )
    p.add_argument(
        "--smoothing-level",
        action="append",
        default=None,
        help="Smoothing level(s) to export (repeatable): raw/30Hz/10Hz (or 30/10).",
    )
    p.add_argument(
        "--target-fs",
        action="append",
        type=float,
        default=None,
        help="Additional numeric target fs values in Hz to export (repeatable).",
    )
    p.add_argument(
        "--edges-path",
        type=Path,
        default=None,
        help=(
            "Optional .npy containing dict[dataset_stem -> edges] with edges.shape==(n_trials,2) in seconds. "
            "If provided, traces (and ap_times) are restricted to these windows before export."
        ),
    )
    p.add_argument(
        "--pad-mode",
        choices=["nan", "edge"],
        default="nan",
        help="How to pad variable-length trials when windowing changes lengths (default: nan).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    globs = args.dataset_glob or ["*.mat"]
    levels = _resolve_levels(args.smoothing_level, args.target_fs)

    if not data_root.exists():
        raise FileNotFoundError(data_root)
    mat_files = list(_iter_mat_files(data_root, globs))
    if not mat_files:
        raise FileNotFoundError(f"No datasets matched under {data_root} with globs={globs}")

    edges_dict: Optional[Dict[str, np.ndarray]] = None
    if args.edges_path is not None:
        edges_dict = _load_edges_dict(args.edges_path.expanduser().resolve())

    exported = 0
    for mat_path in mat_files:
        dataset_tag = mat_path.stem
        if dataset_tag.startswith("biophysd_"):
            dataset_key = dataset_tag[len("biophysd_") :]
        else:
            dataset_key = dataset_tag
        edges = None
        if edges_dict is not None:
            if dataset_key not in edges_dict:
                raise KeyError(f"Dataset {dataset_key!r} not found in edges dict {args.edges_path}")
            edges = np.asarray(edges_dict[dataset_key], dtype=float)

        for label, fs in levels:
            out_path = out_root / label / mat_path.name
            export_one(
                mat_path,
                out_path=out_path,
                target_fs=fs,
                edges=edges,
                pad_mode=str(args.pad_mode),
                overwrite=bool(args.overwrite),
            )
        exported += 1

    print(f"Exported {exported} dataset(s) into {out_root} ({', '.join([l for l, _ in levels])}).")


if __name__ == "__main__":
    main()

