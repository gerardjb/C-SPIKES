#!/usr/bin/env python3
"""
Raincloud plot of trialwise correlations organized by downsample rate.

This reads `results/trialwise_correlations.csv` (produced by `scripts/trialwise_correlations.py`)
and plots a per-method distribution of correlations for each smoothing/downsample label.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("results/trialwise_correlations.csv"), help="Input CSV path.")
    p.add_argument("--out", type=Path, default=Path("results/trialwise_raincloud.png"), help="Output image path.")
    p.add_argument("--corr-sigma-ms", type=float, default=50.0, help="Correlation smoothing sigma (ms) to select.")
    p.add_argument("--method", action="append", default=None, help="Method(s) to plot (repeatable).")
    p.add_argument(
        "--smoothing",
        action="append",
        default=None,
        help="Smoothing/downsample label(s) to include (repeatable; e.g. raw, 30Hz, 10Hz).",
    )
    p.add_argument("--dataset", action="append", default=None, help="Restrict to dataset stem(s). Repeatable.")
    p.add_argument("--run", action="append", default=None, help="Restrict to run tag(s). Repeatable.")
    p.add_argument(
        "--run-by-method",
        action="append",
        default=None,
        help="Method-to-run mapping, e.g. --run-by-method pgas=base --run-by-method ens2=ens2_custom_... (repeatable).",
    )
    p.add_argument(
        "--reduce",
        choices=["trial", "dataset"],
        default="trial",
        help="Sampling unit: trialwise points (trial) or dataset means (dataset).",
    )
    p.add_argument("--title", type=str, default=None, help="Optional figure title.")
    p.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.0), help="Y limits, e.g. --ylim 0 1")
    p.add_argument("--figsize", type=float, nargs=2, default=(7.2, 4.2), help="Figure size in inches.")
    p.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
    p.add_argument(
        "--group-spacing",
        type=float,
        default=1.25,
        help="Spacing between adjacent smoothing groups on the x-axis (default: 1.25).",
    )
    p.add_argument(
        "--method-label-x-offset-frac",
        type=float,
        default=0.05,
        help="Right-hand method label x-offset as a fraction of x-span (default: 0.05).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    from c_spikes.viz.trialwise_plots import plot_raincloud_by_downsample

    fig, _axes = plot_raincloud_by_downsample(
        csv_path=args.csv,
        out_path=args.out,
        corr_sigma_ms=float(args.corr_sigma_ms),
        methods=args.method,
        smoothings=args.smoothing,
        runs=args.run,
        run_by_method=args.run_by_method,
        datasets=args.dataset,
        reduce=str(args.reduce),
        title=args.title,
        ylim=(float(args.ylim[0]), float(args.ylim[1])),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
        seed=int(args.seed),
        group_spacing=float(args.group_spacing),
        method_label_x_offset_frac=float(args.method_label_x_offset_frac),
    )
    fig.savefig(args.out)
    print(f"[plot] Wrote {args.out}")


if __name__ == "__main__":
    main()
