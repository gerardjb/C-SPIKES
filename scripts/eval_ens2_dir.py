#!/usr/bin/env python3
"""
Evaluate a single ENS2 checkpoint directory on a directory of Janelia .mat files.

This runs ENS2-only inference (no PGAS/CASCADE) and reports correlation-to-GT for
each dataset, plus aggregate summary stats.

Example:
  python scripts/eval_ens2_dir.py \
    --ens2-root results/Pretrained_models/ens2_published \
    --dataset-dir data/janelia_8f/excitatory \
    --out-dir results/ens2_eval/ens2_published__excitatory
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from c_spikes.inference.workflow import DatasetRunConfig, MethodSelection, SmoothingLevel, run_inference_for_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ens2-root", type=Path, required=True, help="Directory containing exc_ens2_pub.pt/inh_ens2_pub.pt.")
    p.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing Janelia .mat files.")
    p.add_argument("--pattern", type=str, default="*.mat", help="Glob pattern within dataset-dir (default: *.mat).")
    p.add_argument(
        "--edges-file",
        type=Path,
        default=None,
        help="Optional edges .npy (dict dataset_stem -> edges[n_trials,2]) to restrict correlations to selected windows.",
    )
    p.add_argument("--neuron-type", type=str, default="Exc", choices=["Exc", "exc", "Inh", "inh"], help="ENS2 neuron type.")
    p.add_argument("--corr-sigma-ms", type=float, default=50.0, help="Gaussian sigma (ms) for correlation smoothing.")
    p.add_argument("--smoothing", type=float, default=None, help="Optional pre-inference smoothing target Hz (None=raw).")
    p.add_argument("--no-cache", action="store_true", help="Disable inference cache (force recompute).")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of datasets (for quick runs).")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for summary.json and summary.csv.")
    return p.parse_args()


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    ens2_root = args.ens2_root.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ens2_root.exists():
        raise FileNotFoundError(ens2_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    datasets = sorted(dataset_dir.glob(args.pattern))
    if not datasets:
        raise FileNotFoundError(f"No datasets found in {dataset_dir} matching {args.pattern!r}")
    if args.limit is not None:
        datasets = datasets[: int(args.limit)]

    checkpoint_name = "exc_ens2_pub.pt" if args.neuron_type.lower().startswith("exc") else "inh_ens2_pub.pt"
    checkpoint_path = ens2_root / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    edges_lookup: Optional[Dict[str, Any]] = None
    if args.edges_file is not None:
        edges_path = args.edges_file.expanduser().resolve()
        if not edges_path.exists():
            raise FileNotFoundError(edges_path)
        edges_lookup = np.load(edges_path, allow_pickle=True).item()

    results: List[Dict[str, Any]] = []
    for ds_path in datasets:
        edges = None
        if edges_lookup is not None:
            candidate = edges_lookup.get(ds_path.stem)
            if candidate is not None:
                edges = np.asarray(candidate, dtype=np.float64)
                if edges.ndim != 2 or edges.shape[1] != 2:
                    print(f"[warn] Ignoring invalid edges shape {edges.shape} for {ds_path.stem}")
                    edges = None
        cfg = DatasetRunConfig(
            dataset_path=ds_path,
            neuron_type=args.neuron_type,
            smoothing=SmoothingLevel(target_fs=args.smoothing),
            edges=edges,
            selection=MethodSelection(run_pgas=False, run_ens2=True, run_cascade=False),
            use_cache=not args.no_cache,
            corr_sigma_ms=float(args.corr_sigma_ms),
        )
        outputs = run_inference_for_dataset(
            cfg,
            pgas_constants=Path("parameter_files/constants_GCaMP8_soma.json"),
            pgas_gparam=Path("src/c_spikes/pgas/20230525_gold.dat"),
            pgas_output_root=Path("results/pgas_output/unused"),
            ens2_pretrained_root=ens2_root,
            cascade_model_root=Path("results/Pretrained_models"),
        )
        corr = outputs.get("correlations", {}).get("ens2", float("nan"))
        summary = outputs.get("summary", {})
        results.append(
            {
                "dataset": ds_path.stem,
                "path": str(ds_path),
                "correlation": _safe_float(corr),
                "gt_count": int(summary.get("gt_count", 0) or 0),
            }
        )

    corrs = np.asarray([r["correlation"] for r in results], dtype=float)
    finite = corrs[np.isfinite(corrs)]
    aggregate: Dict[str, Any] = {
        "n_datasets": int(len(results)),
        "n_finite": int(finite.size),
        "mean_correlation": float(np.mean(finite)) if finite.size else float("nan"),
        "median_correlation": float(np.median(finite)) if finite.size else float("nan"),
        "min_correlation": float(np.min(finite)) if finite.size else float("nan"),
        "max_correlation": float(np.max(finite)) if finite.size else float("nan"),
    }

    payload: Dict[str, Any] = {
        "ens2_root": str(ens2_root),
        "checkpoint": str(checkpoint_path),
        "neuron_type": args.neuron_type,
        "dataset_dir": str(dataset_dir),
        "pattern": args.pattern,
        "edges_file": str(args.edges_file) if args.edges_file is not None else None,
        "corr_sigma_ms": float(args.corr_sigma_ms),
        "smoothing_hz": args.smoothing,
        "use_cache": bool(not args.no_cache),
        "aggregate": aggregate,
        "per_dataset": results,
    }

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "correlation", "gt_count", "path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[eval] Wrote: {json_path}")
    print(f"[eval] Wrote: {csv_path}")
    print(
        "[eval] Aggregate: "
        f"n={aggregate['n_datasets']} finite={aggregate['n_finite']} "
        f"mean={aggregate['mean_correlation']:.4f} median={aggregate['median_correlation']:.4f} "
        f"min={aggregate['min_correlation']:.4f} max={aggregate['max_correlation']:.4f}"
    )


if __name__ == "__main__":
    main()
