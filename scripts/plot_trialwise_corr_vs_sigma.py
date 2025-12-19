#!/usr/bin/env python3
"""
Plot mean ± SEM correlation vs correlation smoothing sigma (ms).

This reads the CSV produced by `scripts/trialwise_correlations.py` and produces a
Matlab-like "shaded error bar" plot:
  - x-axis: corr_sigma_ms (ms)
  - y-axis: mean correlation
  - shaded band: ± SEM across datasets (or across trials, if requested)

Example:
  PYTHONPATH=src python scripts/plot_trialwise_corr_vs_sigma.py \
    --csv results/trialwise_correlations.csv \
    --out results/trialwise_corr_vs_sigma.png \
    --run pgasraw --run cascadein_nodisc_ens2 \
    --smoothing raw
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# In some sandboxed/HPC environments Intel OpenMP shared-memory init can fail
# (e.g. "Can't open SHM2"). Using sequential threading avoids that.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ.setdefault("MPLBACKEND", "Agg")
# Ensure matplotlib uses a writable cache/config dir (common issue on HPC).
_default_cache_root = Path.cwd() / "tmp" / "mpl_cache"
try:
    _default_cache_root.mkdir(parents=True, exist_ok=True)
except OSError:
    # Fall back to /tmp if cwd isn't writable.
    _default_cache_root = Path("/tmp") / "c_spikes_mpl_cache"
    _default_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_default_cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(_default_cache_root))

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


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
    p.add_argument("--csv", type=Path, default=Path("results/trialwise_correlations.csv"), help="Input CSV path.")
    p.add_argument("--out", type=Path, default=Path("results/trialwise_corr_vs_sigma.png"), help="Output image path.")
    p.add_argument("--run", action="append", help="Restrict to run tag(s). Repeatable.")
    p.add_argument("--dataset", action="append", help="Restrict to dataset stem(s). Repeatable.")
    p.add_argument("--smoothing", action="append", help="Restrict to smoothing label(s) (raw/30Hz/10Hz). Repeatable.")
    p.add_argument("--method", action="append", help="Restrict to method(s) (pgas/cascade/ens2/...). Repeatable.")
    p.add_argument(
        "--reduce",
        choices=["dataset", "trial"],
        default="dataset",
        help=(
            "What a single sample is for SEM. "
            "`dataset` = compute per-dataset mean first (default; closer to 'per-cell' SEM). "
            "`trial` = treat each trial correlation as a sample."
        ),
    )
    p.add_argument("--title", type=str, default=None, help="Optional figure title.")
    p.add_argument("--ylabel", type=str, default="Pearson correlation", help="Y-axis label.")
    p.add_argument("--ylim", type=float, nargs=2, default=(0.2, 1.0), help="Y limits, e.g. --ylim 0.2 1.0")
    p.add_argument("--figsize", type=float, nargs=2, default=(7.2, 2.8), help="Figure size in inches.")
    p.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    p.add_argument("--legend", action="store_true", help="Use a legend instead of right-side labels.")
    p.add_argument(
        "--right-label-x-offset-frac",
        type=float,
        default=0.08,
        help="Right label x-offset as fraction of x-span (default: 0.08).",
    )
    p.add_argument(
        "--right-label-xlim-frac",
        type=float,
        default=0.22,
        help="Extra xlim to the right as fraction of x-span (default: 0.22).",
    )
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


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _mean_sem(values: Sequence[float]) -> Tuple[float, float, int]:
    arr = _finite(np.asarray(values, dtype=np.float64))
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, n
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean, sem, n


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def _coerce_float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _place_right_labels(
    ax: matplotlib.axes.Axes,
    fig: matplotlib.figure.Figure,
    *,
    x: float,
    y_positions: List[float],
    labels: List[str],
    colors: List[str],
    min_sep: float,
) -> None:
    # Respect input ordering (caller can rank-order labels); apply offsets to avoid overlap.
    ys = [float(y) for y in y_positions]
    labs = list(labels)
    cols = list(colors)

    # First: coarse separation in data coords, pushing downward.
    for i in range(1, len(ys)):
        if not np.isfinite(ys[i - 1]) or not np.isfinite(ys[i]):
            continue
        if ys[i - 1] - ys[i] < min_sep:
            ys[i] = ys[i - 1] - min_sep

    # Second: refine in display coords using bounding boxes (font-size aware).
    texts: List[matplotlib.text.Text] = []
    for y, lab, col in zip(ys, labs, cols):
        texts.append(
            ax.text(
                x,
                y,
                lab,
                color=col,
                fontsize=14,
                fontweight="bold",
                ha="left",
                va="center",
                clip_on=False,
            )
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad_px = 2.0
    prev_bbox = None
    for text in texts:
        bbox = text.get_window_extent(renderer=renderer).expanded(1.0, 1.08)
        while prev_bbox is not None and bbox.overlaps(prev_bbox):
            # Push downward so this label's top clears the previous label's bottom.
            shift_px = float(bbox.y1 - prev_bbox.y0 + pad_px)
            x_disp, y_disp = ax.transData.transform(text.get_position())
            new_y_disp = y_disp - shift_px
            _, new_y_data = ax.transData.inverted().transform((x_disp, new_y_disp))
            text.set_position((x, float(new_y_data)))
            fig.canvas.draw()
            bbox = text.get_window_extent(renderer=renderer).expanded(1.0, 1.08)
        prev_bbox = bbox

    # If labels overflow the axes vertically, shift the whole stack to fit.
    ax_bbox = ax.get_window_extent(renderer=renderer)
    if texts:
        bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
        top = max(b.y1 for b in bboxes)
        bottom = min(b.y0 for b in bboxes)
        shift_up = 0.0
        if top > ax_bbox.y1:
            shift_up = ax_bbox.y1 - top
        if bottom + shift_up < ax_bbox.y0:
            shift_up = ax_bbox.y0 - bottom
        if shift_up != 0.0:
            for t in texts:
                x_disp, y_disp = ax.transData.transform(t.get_position())
                new_y_disp = y_disp + shift_up
                _, new_y_data = ax.transData.inverted().transform((x_disp, new_y_disp))
                t.set_position((x, float(new_y_data)))


def _y_at_x(xs: np.ndarray, ys: np.ndarray, x_target: float) -> float:
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return float("nan")
    if xs.size == 1:
        return float(ys[0])
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.interp(float(x_target), xs, ys))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    from c_spikes.viz.trialwise_plots import plot_corr_vs_sigma

    csv_path = args.csv.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_corr_vs_sigma(
        csv_path=csv_path,
        out_path=out_path,
        runs=args.run,
        datasets=args.dataset,
        smoothings=args.smoothing,
        methods=args.method,
        reduce=str(args.reduce),
        title=args.title,
        ylabel=str(args.ylabel),
        ylim=(float(args.ylim[0]), float(args.ylim[1])),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
        legend=bool(args.legend),
        right_label_x_offset_frac=float(args.right_label_x_offset_frac),
        right_label_xlim_frac=float(args.right_label_xlim_frac),
    )
    print(f"[plot] Wrote {out_path}")


if __name__ == "__main__":
    main()
