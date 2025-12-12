#!/usr/bin/env python3
"""
Summarize and visualize ENS2 sweep evaluation results produced by:
  - scripts/sweep_pgas_to_ens2.sh (training + eval)
  - scripts/eval_ens2_dir.py (eval output)

This script:
  - Scans an eval root for per-model `summary.json` files
  - Builds a CSV leaderboard (mean/median correlation across datasets)
  - Produces coarse heatmaps over (smooth, duty) for each K and spike rate
  - Plots best score vs K (optionally with ens2_published baseline)

Example:
  python scripts/plot_ens2_sweep_results.py \
    --eval-root results/ens2_sweep_eval/test_refactor__excitatory
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


MODEL_RE = re.compile(
    r"k(?P<k>\\d+)_r(?P<rate>\\d+)_s(?P<smooth>[0-9]+p[0-9]+)_d(?P<duty>[0-9]+p[0-9]+)_sb(?P<seed>\\d+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", type=Path, required=True, help="Directory containing per-model eval subdirs.")
    p.add_argument(
        "--metric",
        type=str,
        default="mean_correlation",
        choices=["mean_correlation", "median_correlation"],
        help="Aggregate metric from eval summaries to visualize.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots/CSVs (default: <eval-root>/plots).",
    )
    return p.parse_args()


def _float_from_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_model_params(model_name: str) -> Optional[Dict[str, Any]]:
    m = MODEL_RE.search(model_name)
    if not m:
        return None
    return {
        "k": int(m.group("k")),
        "rate": int(m.group("rate")),
        "smooth": _float_from_tag(m.group("smooth")),
        "duty": _float_from_tag(m.group("duty")),
        "seed_base": int(m.group("seed")),
    }


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    if not eval_root.exists():
        raise FileNotFoundError(eval_root)

    out_dir = (args.out_dir or (eval_root / "plots")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(eval_root.rglob("summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary.json files found under {eval_root}")

    rows: List[Dict[str, Any]] = []
    baseline: Optional[Tuple[str, float]] = None
    for sp in summary_paths:
        model_dir = sp.parent
        model_name = model_dir.name
        payload = _load_summary(sp)
        agg = payload.get("aggregate", {})
        metric_val = float(agg.get(args.metric, float("nan")))

        params = _extract_model_params(model_name)
        row: Dict[str, Any] = {
            "model_name": model_name,
            "summary_path": str(sp),
            "ens2_root": payload.get("ens2_root"),
            "checkpoint": payload.get("checkpoint"),
            "metric": args.metric,
            "score": metric_val,
            "n_datasets": int(agg.get("n_datasets", 0) or 0),
            "n_finite": int(agg.get("n_finite", 0) or 0),
            "mean_correlation": float(agg.get("mean_correlation", float("nan"))),
            "median_correlation": float(agg.get("median_correlation", float("nan"))),
        }
        if params is not None:
            row.update(params)
        else:
            # Treat anything unparsable as a potential baseline (e.g., baseline__ens2_published).
            if model_name.startswith("baseline") and baseline is None:
                baseline = (model_name, metric_val)
        rows.append(row)

    # Write leaderboard CSV
    csv_path = out_dir / "leaderboard.csv"
    fieldnames = [
        "model_name",
        "k",
        "rate",
        "smooth",
        "duty",
        "seed_base",
        "score",
        "mean_correlation",
        "median_correlation",
        "n_datasets",
        "n_finite",
        "ens2_root",
        "checkpoint",
        "summary_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        def _score_for_sort(row: Dict[str, Any]) -> float:
            try:
                score = float(row.get("score", float("nan")))
            except Exception:
                return float("-inf")
            return score if np.isfinite(score) else float("-inf")

        for r in sorted(rows, key=_score_for_sort, reverse=True):
            writer.writerow(r)
    print(f"[viz] Wrote: {csv_path}")

    # Print top-10
    scored = [r for r in rows if np.isfinite(float(r.get("score", float("nan")))) and "k" in r]
    scored_sorted = sorted(scored, key=lambda r: float(r["score"]), reverse=True)
    if scored_sorted:
        print("[viz] Top models:")
        for r in scored_sorted[:10]:
            print(
                f"  {r['model_name']}: {args.metric}={float(r['score']):.4f} "
                f"(k={r.get('k')} rate={r.get('rate')} smooth={r.get('smooth')} duty={r.get('duty')})"
            )
    if baseline is not None:
        print(f"[viz] Baseline {baseline[0]}: {args.metric}={baseline[1]:.4f}")

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[viz] Matplotlib unavailable ({exc}); wrote CSV only.")
        return

    # Build heatmaps per K
    ks = sorted({int(r["k"]) for r in scored_sorted if "k" in r})
    rates = sorted({int(r["rate"]) for r in scored_sorted if "rate" in r})
    smooths = sorted({float(r["smooth"]) for r in scored_sorted if "smooth" in r})
    duties = sorted({float(r["duty"]) for r in scored_sorted if "duty" in r})

    if ks and rates and smooths and duties:
        for k in ks:
            fig, axes = plt.subplots(1, len(rates), figsize=(4 * len(rates), 4), squeeze=False)
            for col, rate in enumerate(rates):
                mat = np.full((len(smooths), len(duties)), np.nan, dtype=float)
                for r in scored_sorted:
                    if int(r.get("k", -1)) != k or int(r.get("rate", -1)) != rate:
                        continue
                    si = smooths.index(float(r["smooth"]))
                    di = duties.index(float(r["duty"]))
                    mat[si, di] = float(r["score"])
                ax = axes[0][col]
                im = ax.imshow(
                    mat,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    vmin=np.nanmin(mat) if np.isfinite(mat).any() else None,
                    vmax=np.nanmax(mat) if np.isfinite(mat).any() else None,
                )
                ax.set_title(f"K={k}, rate={rate} Hz")
                ax.set_xticks(range(len(duties)))
                ax.set_xticklabels([f"{d:.2f}" for d in duties], rotation=45, ha="right")
                ax.set_yticks(range(len(smooths)))
                ax.set_yticklabels([f"{s:.2f}" for s in smooths])
                ax.set_xlabel("duty")
                ax.set_ylabel("smooth (s)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=args.metric)
            fig.suptitle(f"ENS2 sweep heatmaps ({args.metric})")
            fig.tight_layout()
            out_path = out_dir / f"heatmap_k{k}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[viz] Wrote: {out_path}")

    # Best score vs K (+ baseline)
    if ks:
        best_by_k = []
        for k in ks:
            vals = [float(r["score"]) for r in scored_sorted if int(r.get("k", -1)) == k and np.isfinite(r["score"])]
            best_by_k.append(float(np.max(vals)) if vals else float("nan"))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, best_by_k, marker="o")
        ax.set_xlabel("K (number of cell param sets)")
        ax.set_ylabel(args.metric)
        ax.set_title("Best model score vs K")
        if baseline is not None and np.isfinite(baseline[1]):
            ax.axhline(baseline[1], color="k", linestyle="--", linewidth=1, label="ens2_published")
            ax.legend()
        fig.tight_layout()
        out_path = out_dir / "best_vs_k.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[viz] Wrote: {out_path}")


if __name__ == "__main__":
    main()
