#!/usr/bin/env python3
"""
Compute epoch-wise firing-rate statistics for Janelia-format .mat files.

Each input .mat is expected to contain:
  - time_stamps: array shaped (n_epochs, n_samples) in seconds
  - ap_times: 1D array of spike times in seconds (global across epochs)

For every epoch, this script reports:
  - mean firing rate (Hz) within that epoch
  - maximum local firing rate (Hz) over fixed-width bins (default 100 ms)
  - p99 of a smoothed spike-rate trace (Hz)
  - correlation time of the smoothed rate (s; lag where ACF drops below 1/e)
  - duty cycle of the smoothed rate (fraction of time bins above a small threshold)

Spikes are restricted to those that fall within the [start, end] bounds of each epoch.

Example:
  python scripts/epoch_rate_stats.py \
    --data-dir data/janelia_8f/excitatory \
    --bin-width 0.1 \
    --save-csv results/epoch_rate_stats.csv \
    --plot-hist
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d


def _as_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float).ravel()


def compute_epoch_stats(
    epoch_times: np.ndarray,
    spike_times: np.ndarray,
    *,
    bin_width: float = 0.1,
    tolerance: float = 1e-9,
) -> Tuple[float, float, int, float, float]:
    """
    Return (mean_rate_hz, max_local_rate_hz, n_spikes, start, end) for one epoch.
    """
    times = _as_1d(epoch_times)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return 0.0, 0.0, 0, float("nan"), float("nan")

    start = float(times.min())
    end = float(times.max())
    duration = max(end - start, tolerance)

    spikes = _as_1d(spike_times)
    spikes = spikes[np.isfinite(spikes)]
    in_epoch = spikes[(spikes >= start - tolerance) & (spikes <= end + tolerance)]
    n_spikes = int(in_epoch.size)
    mean_rate = n_spikes / duration

    if n_spikes == 0:
        return float(mean_rate), 0.0, n_spikes, start, end

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    bins = np.arange(start, end + bin_width, bin_width)
    counts, _ = np.histogram(in_epoch, bins=bins)
    max_local = float(counts.max() / bin_width) if counts.size else 0.0
    return float(mean_rate), max_local, n_spikes, start, end


def estimate_correlation_time(rate: np.ndarray, dt: float, *, threshold: float = np.exp(-1)) -> float:
    """Estimate correlation time as first lag where normalized ACF <= threshold."""
    x = np.asarray(rate, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    acf = np.correlate(x, x, mode="full")[x.size - 1 :] / denom
    below = np.where(acf <= threshold)[0]
    if below.size == 0:
        return float((acf.size - 1) * dt)
    return float(below[0] * dt)


def compute_latent_metrics(
    in_epoch_spikes: np.ndarray,
    start: float,
    end: float,
    *,
    bin_width: float,
    smooth_sigma_s: float,
    duty_threshold_frac: float,
) -> Tuple[float, float, float]:
    """
    From spikes within an epoch, build a binned spike-rate trace, smooth it,
    then return (p99_rate_hz, corr_time_s, duty_cycle).
    """
    duration = end - start
    if duration <= 0 or bin_width <= 0:
        return 0.0, float("nan"), 0.0
    bins = np.arange(start, end + bin_width, bin_width)
    counts, _ = np.histogram(in_epoch_spikes, bins=bins)
    rate = counts.astype(float) / bin_width
    if smooth_sigma_s > 0:
        sigma_bins = smooth_sigma_s / bin_width
        rate_sm = gaussian_filter1d(rate, sigma=sigma_bins, mode="nearest")
    else:
        rate_sm = rate
    p99_rate = float(np.percentile(rate_sm, 99)) if rate_sm.size else 0.0
    corr_time = estimate_correlation_time(rate_sm, bin_width)
    if p99_rate <= 0:
        duty = 0.0
    else:
        eps = duty_threshold_frac * p99_rate
        duty = float(np.mean(rate_sm > eps))
    return p99_rate, corr_time, duty


def load_mat_fields(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(mat_path)
    if "time_stamps" not in data:
        raise KeyError(f"{mat_path} missing 'time_stamps'")
    if "ap_times" not in data:
        raise KeyError(f"{mat_path} missing 'ap_times'")
    return np.asarray(data["time_stamps"], dtype=float), _as_1d(data["ap_times"])


def iter_mat_files(data_dir: Path, pattern: str) -> List[Path]:
    return sorted(data_dir.glob(pattern))


def summarize_distribution(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return {
        "count": int(v.size),
        "mean": float(np.mean(v)),
        "std": std,
        "min": float(np.min(v)),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "max": float(np.max(v)),
    }


def print_distribution_summary(label: str, values: Sequence[float]) -> None:
    stats = summarize_distribution(values)
    if not stats:
        print(f"{label}: no finite values.")
        return
    print(
        f"{label}: n={stats['count']} "
        f"mean={stats['mean']:.3f} std={stats['std']:.3f} "
        f"min={stats['min']:.3f} p25={stats['p25']:.3f} "
        f"median={stats['median']:.3f} p75={stats['p75']:.3f} "
        f"max={stats['max']:.3f}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/janelia_8f/excitatory"),
        help="Directory containing .mat files.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.mat",
        help="Glob pattern under data-dir.",
    )
    p.add_argument(
        "--bin-width",
        type=float,
        default=0.1,
        help="Bin width (seconds) for max-local rate (default 0.1s = 100 ms).",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files processed.",
    )
    p.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to write a CSV summary.",
    )
    p.add_argument(
        "--plot-hist",
        action="store_true",
        help="If set, display histograms of mean and max-local rates across epochs.",
    )
    p.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Number of bins for histograms (default 50).",
    )
    p.add_argument(
        "--latent-bin-width",
        type=float,
        default=None,
        help="Bin width (seconds) for latent-rate estimation (defaults to --bin-width).",
    )
    p.add_argument(
        "--latent-smooth-sigma",
        type=float,
        default=0.5,
        help="Gaussian sigma (seconds) to smooth latent spike-rate (default 0.5s).",
    )
    p.add_argument(
        "--duty-threshold-frac",
        type=float,
        default=0.05,
        help="Duty-cycle threshold as fraction of latent p99 rate (default 0.05).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    mat_paths = iter_mat_files(data_dir, args.pattern)
    if args.max_files is not None:
        mat_paths = mat_paths[: int(args.max_files)]
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files matched {args.pattern} under {data_dir}")

    rows: List[Tuple[str, int, float, float, float, int, float, float, float, float, float]] = []
    mean_rates: List[float] = []
    max_local_rates: List[float] = []
    p99_rates: List[float] = []
    corr_times: List[float] = []
    duty_cycles: List[float] = []

    for mat_path in mat_paths:
        try:
            time_stamps, ap_times = load_mat_fields(mat_path)
        except Exception as exc:
            print(f"[WARN] Skipping {mat_path.name}: {exc}")
            continue

        # Normalize time_stamps to shape (n_epochs, n_samples)
        if time_stamps.ndim == 1:
            time_stamps = time_stamps[None, :]

        n_epochs = time_stamps.shape[0]
        print(f"{mat_path.name} ({n_epochs} epochs)")

        for epoch_idx in range(n_epochs):
            epoch_times = time_stamps[epoch_idx]
            mean_rate, max_local, n_spikes, start, end = compute_epoch_stats(
                epoch_times,
                ap_times,
                bin_width=float(args.bin_width),
            )
            spikes = _as_1d(ap_times)
            spikes = spikes[np.isfinite(spikes)]
            in_epoch = spikes[(spikes >= start) & (spikes <= end)] if np.isfinite(start) and np.isfinite(end) else np.array([])
            latent_bin = float(args.latent_bin_width) if args.latent_bin_width is not None else float(args.bin_width)
            p99_rate, corr_time, duty = compute_latent_metrics(
                in_epoch,
                start,
                end,
                bin_width=latent_bin,
                smooth_sigma_s=float(args.latent_smooth_sigma),
                duty_threshold_frac=float(args.duty_threshold_frac),
            )
            duration = float(end - start) if np.isfinite(start) and np.isfinite(end) else float("nan")
            print(
                f"  epoch {epoch_idx:02d}: "
                f"start={start:.3f}s end={end:.3f}s dur={duration:.3f}s "
                f"n_spikes={n_spikes:4d} mean={mean_rate:.3f}Hz max_local={max_local:.3f}Hz "
                f"p99_rate={p99_rate:.3f}Hz corr_time={corr_time:.3f}s duty={duty:.3f}"
            )
            mean_rates.append(mean_rate)
            max_local_rates.append(max_local)
            p99_rates.append(p99_rate)
            corr_times.append(corr_time)
            duty_cycles.append(duty)
            rows.append(
                (
                    mat_path.name,
                    epoch_idx,
                    start,
                    end,
                    duration,
                    n_spikes,
                    mean_rate,
                    max_local,
                    p99_rate,
                    corr_time,
                    duty,
                )
            )

    print("\nDistribution summaries across all epochs:")
    print_distribution_summary("Mean firing rate (Hz)", mean_rates)
    print_distribution_summary(
        f"Max local rate (Hz; {args.bin_width*1000:.0f} ms bins)", max_local_rates
    )
    print_distribution_summary("Latent p99 rate (Hz)", p99_rates)
    print_distribution_summary("Latent correlation time (s)", corr_times)
    print_distribution_summary("Latent duty cycle (fraction)", duty_cycles)

    if args.plot_hist and mean_rates:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.ravel()

        axes[0].hist(mean_rates, bins=int(args.hist_bins), color="tab:blue", alpha=0.8)
        axes[0].set_title("Mean rate per epoch")
        axes[0].set_xlabel("Hz")
        axes[0].set_ylabel("Count")

        axes[1].hist(max_local_rates, bins=int(args.hist_bins), color="tab:orange", alpha=0.8)
        axes[1].set_title(f"Max local rate ({args.bin_width*1000:.0f} ms bins)")
        axes[1].set_xlabel("Hz")
        axes[1].set_ylabel("Count")

        axes[2].hist(p99_rates, bins=int(args.hist_bins), color="tab:green", alpha=0.8)
        axes[2].set_title("Latent p99 rate")
        axes[2].set_xlabel("Hz")
        axes[2].set_ylabel("Count")

        axes[3].hist(corr_times, bins=int(args.hist_bins), color="tab:red", alpha=0.8)
        axes[3].set_title("Latent corr time")
        axes[3].set_xlabel("s")
        axes[3].set_ylabel("Count")

        axes[4].hist(duty_cycles, bins=int(args.hist_bins), color="tab:purple", alpha=0.8)
        axes[4].set_title("Latent duty cycle")
        axes[4].set_xlabel("fraction")
        axes[4].set_ylabel("Count")

        axes[5].axis("off")

        plt.tight_layout()
        plt.show()

    if args.save_csv is not None and rows:
        out_path = args.save_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "file,epoch,start_s,end_s,duration_s,n_spikes,mean_rate_hz,max_local_rate_hz,"
            "latent_p99_rate_hz,latent_corr_time_s,latent_duty_cycle"
        )
        arr = np.asarray(rows, dtype=object)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            for r in arr:
                fh.write(
                    f"{r[0]},{int(r[1])},{float(r[2]):.9g},{float(r[3]):.9g},{float(r[4]):.9g},"
                    f"{int(r[5])},{float(r[6]):.9g},{float(r[7]):.9g},"
                    f"{float(r[8]):.9g},{float(r[9]):.9g},{float(r[10]):.9g}\n"
                )
        print(f"Wrote CSV summary to {out_path}")


if __name__ == "__main__":
    main()
