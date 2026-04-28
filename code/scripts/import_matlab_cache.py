#!/usr/bin/env python3
"""
Import MATLAB-generated method outputs into C-SPIKES' cache + evaluation layout.

This is a "backdoor" utility for bringing externally generated spike-probability traces
into the existing plotting/evaluation workflow without re-running inference.

What it writes:
  - results/inference_cache/<method>/<cache_tag>/<cache_key>.mat|.json
  - results/full_evaluation/<run_tag>/<dataset>/<smoothing>/{comparison,summary}.json

Optional:
  - Merge an externally produced trialwise_correlations.csv into results/trialwise_correlations.csv

Example:
  PYTHONPATH=src python scripts/import_matlab_cache.py \\
    --pred-path /path/to/mlspike_out.mat \\
    --dataset jGCaMP8m_ANM472179_cell02 \\
    --smoothing raw \\
    --method mlspike \\
    --run-tag matlab_mlspike \\
    --data-root data/janelia_8f/excitatory

Directory import (one file per dataset; dataset inferred from filename by default):
  PYTHONPATH=src python scripts/import_matlab_cache.py \\
    --pred-dir /path/to/preds \\
    --pred-glob '*.mat' --pred-glob '*.npz' \\
    --dataset-regex '(jGCaMP8[fm]_ANM\\d+_cell\\d+)' \\
    --smoothing raw --method mlspike --run-tag matlab_mlspike \\
    --data-root data/janelia_8m/excitatory
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _parse_scalar(value: str) -> Any:
    token = str(value).strip()
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if token.lower() in {"none", "null"}:
        return None
    # Best-effort JSON (numbers, lists, dicts, quoted strings).
    try:
        return json.loads(token)
    except Exception:
        return token


def _parse_kv(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not items:
        return out
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        out[k] = _parse_scalar(v)
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    pred_group = p.add_mutually_exclusive_group(required=True)
    pred_group.add_argument(
        "--pred-path",
        type=Path,
        help="Prediction file (.mat or .npz) containing time_stamps + spike_prob.",
    )
    pred_group.add_argument(
        "--pred-dir",
        type=Path,
        help="Directory containing many prediction files (one per dataset).",
    )
    p.add_argument(
        "--pred-glob",
        action="append",
        default=None,
        help=(
            "Glob(s) under --pred-dir to select files (repeatable). "
            "Default: '*.mat' and '*.npz'."
        ),
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Dataset stem (without .mat). Required when using --pred-path.",
    )
    p.add_argument(
        "--dataset-regex",
        default=None,
        help=(
            "Regex to extract dataset stem from each filename when using --pred-dir. "
            "If the pattern has a capture group, group(1) is used; otherwise the full match. "
            "Default: use file stem (filename without extension)."
        ),
    )
    p.add_argument("--smoothing", default="raw", help="Smoothing label used in full_evaluation (raw/30Hz/10Hz).")
    p.add_argument("--method", default="matlab", help="Method name to store under results/inference_cache/<method>/...")
    p.add_argument("--label", default=None, help="User-facing label stored in comparison.json (default: same as --method).")
    p.add_argument("--run-tag", required=True, help="Run tag directory under results/full_evaluation/<run-tag>/...")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional root containing <dataset>.mat to compute a stable trace_hash (recommended).",
    )
    p.add_argument("--eval-root", type=Path, default=Path("results/full_evaluation"), help="Full-evaluation root to write into.")
    p.add_argument("--cache-root", type=Path, default=Path("results/inference_cache"), help="Inference cache root to write into.")
    p.add_argument("--cache-tag", default=None, help="Cache subdir under method (default: dataset stem).")
    p.add_argument("--time-key", default="time_stamps", help="Variable name for time stamps in pred file.")
    p.add_argument("--spike-prob-key", default="spike_prob", help="Variable name for spike probabilities in pred file.")
    p.add_argument("--reconstruction-key", default="reconstruction", help="Optional variable name for reconstruction (set to empty to ignore).")
    p.add_argument("--discrete-spikes-key", default="discrete_spikes", help="Optional variable name for discrete spikes (set to empty to ignore).")
    p.add_argument("--corr-sigma-ms", type=float, default=50.0, help="corr_sigma_ms placeholder stored in summary.json.")
    p.add_argument("--config-json", type=Path, default=None, help="Optional JSON file containing a config dict for cache-keying.")
    p.add_argument(
        "--config-kv",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra config entries (repeatable). Values are parsed as JSON when possible.",
    )
    p.add_argument(
        "--trialwise-in-csv",
        type=Path,
        default=None,
        help="Optional trialwise_correlations-style CSV to merge in (schema must match).",
    )
    p.add_argument(
        "--trialwise-out-csv",
        type=Path,
        default=Path("results/trialwise_correlations.csv"),
        help="Where to write merged trialwise correlations CSV.",
    )
    p.add_argument(
        "--trialwise-on-conflict",
        choices=["replace", "keep"],
        default="replace",
        help="When merging trialwise CSV rows, replace or keep existing rows (default: replace).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    from c_spikes.inference.import_external import import_external_method, merge_trialwise_correlations_csv

    config_base: Dict[str, Any]
    if args.config_json is not None:
        cfg_path = args.config_json.expanduser().resolve()
        obj = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"--config-json must contain a JSON object/dict; got {type(obj).__name__}")
        config_base = obj
    else:
        config_base = {"source": "matlab"}
    config_base.update(_parse_kv(args.config_kv))

    reconstruction_key = str(args.reconstruction_key).strip() or None
    discrete_key = str(args.discrete_spikes_key).strip() or None

    pred_paths: list[Path] = []
    if args.pred_path is not None:
        if not args.dataset:
            raise ValueError("--dataset is required when using --pred-path.")
        pred_paths = [Path(args.pred_path)]
    else:
        if args.cache_tag is not None:
            raise ValueError("--cache-tag is not supported with --pred-dir (cache_tag is derived per dataset).")
        pred_dir = Path(args.pred_dir).expanduser().resolve()
        if not pred_dir.exists() or not pred_dir.is_dir():
            raise FileNotFoundError(pred_dir)
        globs = args.pred_glob or ["*.mat", "*.npz"]
        for pattern in globs:
            pred_paths.extend(sorted(pred_dir.glob(str(pattern))))
        # Dedup while preserving sort order.
        seen: set[Path] = set()
        pred_paths = [p for p in pred_paths if not (p in seen or seen.add(p))]
        if not pred_paths:
            raise FileNotFoundError(f"No prediction files found under {pred_dir} for globs={globs!r}")

    dataset_re: Optional[re.Pattern[str]] = None
    if args.dataset_regex:
        dataset_re = re.compile(str(args.dataset_regex))
    default_dataset_re = re.compile(r"(jGCaMP[0-9A-Za-z]+_ANM\d+_cell\d+)")

    imported: list[Dict[str, Any]] = []
    for pred_path in pred_paths:
        pred_path = pred_path.expanduser().resolve()
        if args.pred_path is not None:
            dataset = str(args.dataset).strip()
        else:
            name = pred_path.name
            if dataset_re is None:
                dataset = pred_path.stem
                if args.data_root is not None:
                    data_root = Path(args.data_root).expanduser().resolve()
                    expected = data_root / f"{dataset}.mat"
                    if not expected.exists():
                        m2 = default_dataset_re.search(name)
                        if m2:
                            candidate = m2.group(1)
                            if (data_root / f"{candidate}.mat").exists():
                                dataset = candidate
            else:
                m = dataset_re.search(name)
                if not m:
                    raise ValueError(f"--dataset-regex did not match filename: {name}")
                dataset = m.group(1) if m.groups() else m.group(0)
        dataset = str(dataset).strip()
        if not dataset:
            raise ValueError(f"Could not infer a non-empty dataset stem for pred file: {pred_path}")

        config = dict(config_base)
        config.setdefault("pred_file", pred_path.name)

        meta = import_external_method(
            pred_path=pred_path,
            method=str(args.method),
            dataset=dataset,
            smoothing=str(args.smoothing),
            run_tag=str(args.run_tag),
            data_root=args.data_root,
            eval_root=args.eval_root,
            cache_root=args.cache_root,
            cache_tag=dataset,  # per-dataset cache dir
            label=args.label,
            config=config,
            time_key=str(args.time_key),
            spike_prob_key=str(args.spike_prob_key),
            reconstruction_key=reconstruction_key,
            discrete_spikes_key=discrete_key,
            corr_sigma_ms=float(args.corr_sigma_ms),
        )
        imported.append(meta)
        print(
            "[import] "
            f"method={meta['method']} dataset={meta['dataset']} smoothing={meta['smoothing']} run_tag={meta['run_tag']} "
            f"cache_tag={meta['cache_tag']} cache_key={meta['cache_key']} fs={meta['sampling_rate']:.3f}"
        )

    if imported:
        print(f"[import] Imported {len(imported)} file(s).")
        print("[import] Note: correlations are not computed by this importer; run `python -m c_spikes.cli.run --eval-only` to fill summary.json correlations.")

    if args.trialwise_in_csv is not None:
        out_csv = merge_trialwise_correlations_csv(
            base_csv=args.trialwise_out_csv,
            incoming_csv=args.trialwise_in_csv,
            out_csv=args.trialwise_out_csv,
            on_conflict=str(args.trialwise_on_conflict),
        )
        print(f"[import] trialwise CSV: {out_csv}")


if __name__ == "__main__":
    main()
