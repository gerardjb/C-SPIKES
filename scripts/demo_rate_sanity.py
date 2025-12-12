#!/usr/bin/env python3
"""
Sanity check: compare spike-rate statistics between
  1) a synthetic dataset produced by syn_gen (CAttached .mat files)
  2) the original ground-truth dataset used for PGAS inference.

For each, report:
  - Mean spike rate (Hz)
  - Maximum local spike rate over 100 ms bins (Hz)

Example:
  python scripts/demo_rate_sanity.py \
    --synthetic-dir results/Ground_truth/synth_jGCaMP8f_ANM471993_cell01_synth \
    --gt-mat gt_data/jGCaMP8f_ANM471993_cell03.mat \
    --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio

from c_spikes.utils import load_Janelia_data


def compute_rate_stats(times: np.ndarray, spike_times: np.ndarray, bin_width: float = 0.1) -> Tuple[float, float]:
    """Return (mean_rate_Hz, max_local_rate_Hz) given spike times and observation window."""
    times = np.asarray(times, dtype=float)
    spike_times = np.asarray(spike_times, dtype=float)
    if times.size == 0:
        return 0.0, 0.0
    start, end = times.min(), times.max()
    duration = max(end - start, 1e-9)
    mean_rate = spike_times.size / duration
    # Local rate: histogram with given bin width
    bins = np.arange(start, end + bin_width, bin_width)
    counts, _ = np.histogram(spike_times, bins=bins)
    max_local_rate = counts.max() / bin_width if counts.size else 0.0
    return float(mean_rate), float(max_local_rate)


def stats_from_synthetic_dir(synth_dir: Path, bin_width: float) -> Tuple[float, float]:
    """Aggregate stats across all CAttached files in a synthetic directory.

    Each synthetic CAttached file contains its own local timebase (typically
    starting near zero). Concatenating times across files would therefore
    underestimate the total observation window and inflate mean/max rates.
    Instead, compute stats per trace and aggregate durations.
    """
    total_spikes = 0
    total_duration = 0.0
    max_local_rate = 0.0
    for mat_path in sorted(synth_dir.glob("*.mat")):
        data = sio.loadmat(mat_path)
        if "CAttached" not in data:
            continue
        cattached = data["CAttached"]
        for idx in range(cattached.shape[1]):
            entry = cattached[0, idx]
            names = entry.dtype.names
            if "events_AP" not in names or "fluo_time" not in names:
                continue
            spikes_ap = np.asarray(entry["events_AP"][0, 0], dtype=float).ravel()
            fluo_time = np.asarray(entry["fluo_time"][0, 0], dtype=float).ravel()
            # Convert events_AP from samples (1e4 Hz) to seconds
            spikes_sec = spikes_ap / 1e4
            fluo_time = fluo_time[np.isfinite(fluo_time)]
            spikes_sec = spikes_sec[np.isfinite(spikes_sec)]
            if fluo_time.size == 0:
                continue
            duration = float(fluo_time.max() - fluo_time.min())
            duration = max(duration, 1e-9)
            total_duration += duration
            total_spikes += int(spikes_sec.size)
            _, local_max = compute_rate_stats(fluo_time, spikes_sec, bin_width=bin_width)
            if local_max > max_local_rate:
                max_local_rate = local_max
    if total_duration <= 0:
        return 0.0, 0.0
    mean_rate = total_spikes / total_duration
    return float(mean_rate), float(max_local_rate)


def stats_from_gt_mat(gt_mat: Path, bin_width: float) -> Tuple[float, float]:
    """Compute stats from a GT .mat (time_stamps + spike_times via load_Janelia_data)."""
    time_stamps, _, spike_times = load_Janelia_data(str(gt_mat))
    spike_times = np.asarray(spike_times, dtype=float).ravel()
    # Observation window from time_stamps
    times = np.asarray(time_stamps, dtype=float).ravel()
    return compute_rate_stats(times, spike_times, bin_width=bin_width)


def pick_random_synthetic_trace(synth_dir: Path, seed: int = 0):
    """Pick one synthetic file and return (path, time, dff, spikes)."""
    files = sorted(synth_dir.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No synthetic files found in {synth_dir}")
    rng = np.random.RandomState(seed)
    mat_path = files[int(rng.randint(0, len(files)))]
    data = sio.loadmat(mat_path)
    if "CAttached" not in data:
        raise ValueError(f"CAttached not in {mat_path}")
    entry = data["CAttached"][0, 0]
    names = entry.dtype.names
    if "events_AP" not in names or "fluo_time" not in names or "fluo_mean" not in names:
        raise ValueError(f"Missing fields in {mat_path}")
    spikes_ap = np.asarray(entry["events_AP"][0, 0], dtype=float).ravel()
    spikes_sec = spikes_ap / 1e4
    time = np.asarray(entry["fluo_time"][0, 0], dtype=float).ravel()
    fluo_mean_field = entry["fluo_mean"][0, 0]
    if isinstance(fluo_mean_field, np.ndarray):
        dff = np.asarray(fluo_mean_field).ravel()
    else:
        dff = np.array(fluo_mean_field).ravel()
    return mat_path, time, dff, spikes_sec


def pick_gt_trace(gt_mat: Path, seed: int = 0):
    """Pick a trial from GT data."""
    time_stamps, dff, spike_times = load_Janelia_data(str(gt_mat))
    n_trials = time_stamps.shape[0]
    rng = np.random.RandomState(seed)
    idx = int(rng.randint(0, n_trials))
    time = np.asarray(time_stamps[idx], dtype=float).ravel()
    dff_trace = np.asarray(dff[idx], dtype=float).ravel()
    spikes = np.asarray(spike_times, dtype=float).ravel()
    mask = (spikes >= time.min()) & (spikes <= time.max())
    spikes = spikes[mask]
    return idx, time, dff_trace, spikes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-dir", type=Path, required=True, help="Path to synthetic Ground_truth/synth_* directory.")
    p.add_argument("--gt-mat", type=Path, required=True, help="Ground-truth .mat file used for PGAS inference.")
    p.add_argument("--bin-width", type=float, default=0.1, help="Bin width (seconds) for local rate (default: 0.1s).")
    p.add_argument("--plot", action="store_true", help="Plot a random synthetic file vs a random GT trial.")
    p.add_argument("--plot-seed", type=int, default=0, help="Seed for selecting traces when plotting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bin_width = float(args.bin_width)

    synth_mean, synth_max = stats_from_synthetic_dir(args.synthetic_dir, bin_width)
    gt_mean, gt_max = stats_from_gt_mat(args.gt_mat, bin_width)

    print(f"Synthetic ({args.synthetic_dir}): mean={synth_mean:.3f} Hz, max_local={synth_max:.3f} Hz (bin {bin_width*1000:.0f} ms)")
    print(f"GT ({args.gt_mat.name}):         mean={gt_mean:.3f} Hz, max_local={gt_max:.3f} Hz (bin {bin_width*1000:.0f} ms)")

    if args.plot:
        import matplotlib.pyplot as plt

        mat_path, syn_time, syn_dff, syn_spikes = pick_random_synthetic_trace(args.synthetic_dir, seed=args.plot_seed)
        gt_idx, gt_time, gt_dff, gt_spikes = pick_gt_trace(args.gt_mat, seed=args.plot_seed)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

        axes[0].plot(syn_time, syn_dff, color="tab:blue", label="synthetic dFF")
        if syn_spikes.size:
            axes[0].vlines(syn_spikes, ymin=np.nanmin(syn_dff), ymax=np.nanmax(syn_dff), color="tab:red", alpha=0.3, label="spikes")
        axes[0].set_title(f"Synthetic\n{mat_path.name}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("dFF")
        axes[0].legend()

        axes[1].plot(gt_time, gt_dff, color="tab:green", label="GT dFF")
        if gt_spikes.size:
            axes[1].vlines(gt_spikes, ymin=np.nanmin(gt_dff), ymax=np.nanmax(gt_dff), color="tab:red", alpha=0.3, label="spikes")
        axes[1].set_title(f"GT trial idx {gt_idx}\n{args.gt_mat.name}")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
