#!/usr/bin/env python3
"""
Dot plots of sweep results: median correlation vs number of training cells (K).

This script scans ENS2 sweep evaluation outputs and produces, for each
downsample setting (raw / 10Hz / 30Hz / ...), a dot plot where:
  - x = K (number of cell parameter sets used for synthetic training)
  - y = aggregate median correlation across datasets for that model
  - each dot = one trained model evaluation run
  - a dotted horizontal line = ens2_published median correlation baseline

Typical usage:
  python scripts/plot_ens2_sweep_k_dotplots.py \
    --eval-root results/ens2_sweep_eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# In some sandboxed/HPC environments Intel OpenMP shared-memory init can fail
# (e.g. "Can't open SHM2"). Using sequential threading avoids that.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


MODEL_RE = re.compile(
    r"k(?P<k>\d+)_r(?P<rate>\d+)_s(?P<smooth>[0-9]+p[0-9]+)_d(?P<duty>[0-9]+p[0-9]+)_sb(?P<seed>\d+)"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path("results/ens2_sweep_eval"),
        help=(
            "Either a sweep root (e.g. results/ens2_sweep_eval) containing run dirs, "
            "or a single run dir containing downsample subdirs (raw/10Hz/...)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <eval-root>/plots).",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.10,
        help="Horizontal jitter magnitude for dot plots (default: 0.10).",
    )
    return p.parse_args(argv)


def _is_downsample_tag(name: str) -> bool:
    token = name.strip()
    if token.lower() == "raw":
        return True
    return bool(re.fullmatch(r"\d+(?:p\d+)?Hz", token))


def _has_downsample_children(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_dir() and _is_downsample_tag(child.name):
                return True
    except OSError:
        return False
    return False


def _find_run_roots(eval_root: Path) -> List[Path]:
    eval_root = eval_root.expanduser().resolve()
    if _has_downsample_children(eval_root):
        return [eval_root]
    run_roots = [p for p in eval_root.iterdir() if p.is_dir() and _has_downsample_children(p)]
    return sorted(run_roots)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_k(model_name: str) -> Optional[int]:
    m = MODEL_RE.search(model_name)
    if not m:
        return None
    return int(m.group("k"))


def _iter_downsample_dirs(run_root: Path) -> Iterable[Path]:
    for child in sorted(run_root.iterdir()):
        if child.is_dir() and _is_downsample_tag(child.name):
            yield child


def _expected_smoothing_hz(ds_tag: str) -> Optional[float]:
    token = ds_tag.strip()
    if token.lower() == "raw":
        return None
    if not token.endswith("Hz"):
        return None
    numeric = token[: -len("Hz")].replace("p", ".")
    try:
        return float(numeric)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    if not eval_root.exists():
        raise FileNotFoundError(eval_root)

    out_dir = (args.out_dir or (eval_root / "plots")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid matplotlib cache issues in read-only/home-restricted environments.
    mpl_config_dir = out_dir / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_roots = _find_run_roots(eval_root)
    if not run_roots:
        raise FileNotFoundError(
            f"No sweep run roots found under {eval_root} (expected <run>/{'{raw,10Hz,30Hz}'}/<model>/summary.json)."
        )

    rng = np.random.default_rng(0)
    for run_root in run_roots:
        for ds_dir in _iter_downsample_dirs(run_root):
            summary_paths = sorted(ds_dir.glob("*/summary.json"))
            if not summary_paths:
                continue

            baseline_median: Optional[float] = None
            baseline_payload: Optional[Dict[str, Any]] = None
            points: List[Tuple[int, float, str]] = []

            for sp in summary_paths:
                model_name = sp.parent.name
                payload = _load_json(sp)
                median = float(payload.get("aggregate", {}).get("median_correlation", float("nan")))
                if not np.isfinite(median):
                    continue

                if model_name == "baseline_ens2_published":
                    baseline_median = median
                    baseline_payload = payload
                    continue

                k_val = _extract_k(model_name)
                if k_val is None:
                    continue
                points.append((k_val, median, model_name))

            if not points:
                continue

            expected = _expected_smoothing_hz(ds_dir.name)
            if baseline_median is not None and baseline_payload is not None:
                actual = baseline_payload.get("smoothing_hz", None)
                try:
                    actual_val = None if actual is None else float(actual)
                except Exception:
                    actual_val = None
                ok = (expected is None and actual_val is None) or (
                    expected is not None and actual_val is not None and np.isclose(expected, actual_val, atol=1e-6)
                )
                if not ok:
                    print(
                        f"[warn] baseline_ens2_published under {ds_dir} has smoothing_hz={actual!r} "
                        f"(expected {expected!r}); skipping baseline line."
                    )
                    baseline_median = None

            points.sort(key=lambda row: (row[0], row[2]))
            ks = np.asarray([p[0] for p in points], dtype=float)
            ys = np.asarray([p[1] for p in points], dtype=float)

            jitter = float(max(args.jitter, 0.0))
            if jitter > 0 and ks.size > 1:
                ks_plot = ks + rng.uniform(-jitter, jitter, size=ks.shape)
            else:
                ks_plot = ks

            k_min = int(np.min(ks))
            k_max = int(np.max(ks))
            xticks = list(range(k_min, k_max + 1))

            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.scatter(ks_plot, ys, s=22, alpha=0.85, color="tab:blue", edgecolors="none")

            if baseline_median is not None and np.isfinite(baseline_median):
                ax.axhline(
                    baseline_median,
                    color="k",
                    linestyle=":",
                    linewidth=1.6,
                    label=f"ens2_published median={baseline_median:.3f}",
                )
                ax.legend(loc="lower right", frameon=False)

            ax.set_title(f"{run_root.name} — {ds_dir.name} — median correlation by K")
            ax.set_xlabel("Number of cell parameter sets (K)")
            ax.set_ylabel("Median correlation (ENS2 vs GT)")
            ax.set_xlim(max(0.5, k_min - 0.5), k_max + 0.5)
            ax.set_xticks(xticks)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, axis="y", alpha=0.25)

            out_path = out_dir / f"k_dotplot__{run_root.name}__{ds_dir.name}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"[plot] Wrote: {out_path}")


if __name__ == "__main__":
    main()
