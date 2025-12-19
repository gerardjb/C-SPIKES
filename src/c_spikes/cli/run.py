#!/usr/bin/env python3
"""
CLI wrapper around the c_spikes.pipeline.run_batch orchestration.

Example usages:
  # cascade-only on first dataset
  python -m c_spikes.cli.run --method cascade --max-datasets 1

  # explicit dataset list with PGAS sub-stepping
  python -m c_spikes.cli.run --dataset jGCaMP8f_ANM471993_cell01 \
    --smoothing-level 10Hz --method pgas --pgas-substeps-per-frame 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from c_spikes.pipeline import RunConfig, run_batch
from c_spikes.inference.pgas import PGAS_BM_SIGMA_DEFAULT


def _parse_dataset_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return lines


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    token = value.strip().lower()
    if token in {"none", "null", "auto", "estimate", "estimated"}:
        return None
    return float(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/janelia_8f/excitatory"))
    parser.add_argument("--dataset", action="append", metavar="TAG", help="Dataset stem (without .mat). Repeatable.")
    parser.add_argument("--dataset-glob", type=str, default="*.mat", help="Glob under data-root when --dataset is omitted.")
    parser.add_argument("--dataset-list", type=Path, help="File containing dataset stems (one per line).")
    parser.add_argument("--max-datasets", type=int, help="Limit number of datasets processed.")
    parser.add_argument("--smoothing-level", action="append", metavar="LEVEL", help="raw, 30Hz, 10Hz (repeatable).")
    parser.add_argument("--method", action="append", metavar="NAME", help="Methods to run: pgas, ens2, cascade. Default: all.")
    parser.add_argument("--output-root", type=Path, default=Path("results/full_evaluation"), help="Where to write summaries/manifests.")
    parser.add_argument("--edges-path", type=Path, default=Path("results/excitatory_time_stamp_edges.npy"))
    parser.add_argument("--neuron-type", type=str, default="Exc", help="ENS2 neuron type (Exc or Inh).")
    parser.add_argument(
        "--ens2-pretrained-root",
        type=Path,
        default=Path("results/Pretrained_models/ens2_published"),
        help="ENS2 checkpoint directory (published or custom).",
    )
    parser.add_argument(
        "--ens2-model-tag",
        type=str,
        default=None,
        help=(
            "Resolve a custom ENS2 model directory by run_tag in ens2_manifest.json "
            "(matches training.run_tag or any synthetic_entries.run_tag)."
        ),
    )
    parser.add_argument(
        "--ens2-model-root",
        action="append",
        type=Path,
        default=None,
        help="Root(s) to search for custom ENS2 models when using --ens2-model-tag (default: results/Pretrained_models).",
    )
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached method outputs when available.")
    parser.add_argument("--first-trial-only", action="store_true", help="Restrict processing to the first trial/window.")
    parser.add_argument("--bm-sigma-spike-gap", type=float, default=0.15, help="Gap around spikes when estimating PGAS bm_sigma.")
    parser.add_argument(
        "--pgas-constants",
        type=Path,
        default=Path("parameter_files/constants_GCaMP8_soma.json"),
        help="PGAS base constants JSON (sensor-specific).",
    )
    parser.add_argument(
        "--pgas-gparam",
        type=Path,
        default=Path("src/c_spikes/pgas/20230525_gold.dat"),
        help="PGAS GCaMP parameter file (sensor-specific).",
    )
    parser.add_argument(
        "--pgas-output-root",
        type=Path,
        default=Path("results/pgas_output/comparison"),
        help="Where PGAS writes its output files (traj/param_samples).",
    )
    parser.add_argument(
        "--pgas-bm-sigma",
        type=str,
        default=str(PGAS_BM_SIGMA_DEFAULT),
        help="Fixed PGAS bm_sigma value, or 'auto' to estimate from data (default: fixed).",
    )
    parser.add_argument("--pgas-resample-fs", type=float, help="PGAS resample frequency (Hz). (deprecated, kept for compatibility)")
    parser.add_argument(
        "--cascade-resample-fs",
        type=float,
        default=None,
        help="CASCADE input resample frequency (Hz). Default: None (use input sampling rate).",
    )
    parser.add_argument(
        "--cascade-no-discrete",
        action="store_true",
        help="Skip CASCADE discrete-spike inference (avoids slow/hanging discretization; correlations still computed from spike_prob).",
    )
    parser.add_argument(
        "--cascade-model-root",
        type=Path,
        default=Path("results/Pretrained_models"),
        help="Root directory containing CASCADE pretrained models.",
    )
    parser.add_argument(
        "--trialwise-correlations",
        action="store_true",
        help="Also compute and store per-trial correlations in each summary.json.",
    )
    parser.add_argument("--pgas-maxspikes", type=int, help="PGAS maxspikes override.")
    parser.add_argument("--pgas-c0-first-y", action="store_true", help="Initialize PGAS C0 to first observation.")
    parser.add_argument("--run-tag", type=str, help="Optional run-tag override for output directory naming.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    dataset_stems: Optional[List[str]] = None
    if args.dataset or args.dataset_list:
        dataset_stems = []
        if args.dataset:
            dataset_stems.extend(args.dataset)
        if args.dataset_list:
            dataset_stems.extend(_parse_dataset_list(args.dataset_list))
    methods = args.method if args.method else ("pgas", "ens2", "cascade")

    ens2_pretrained_root = args.ens2_pretrained_root
    if args.ens2_model_tag:
        from c_spikes.ens2.manifest import resolve_model_dir_by_run_tag

        search_roots = args.ens2_model_root or [Path("results/Pretrained_models")]
        ens2_pretrained_root = resolve_model_dir_by_run_tag(args.ens2_model_tag, search_roots)

    cfg = RunConfig(
        data_root=args.data_root,
        dataset_glob=args.dataset_glob,
        datasets=dataset_stems,
        max_datasets=args.max_datasets,
        smoothing_levels=args.smoothing_level,
        output_root=args.output_root,
        edges_path=args.edges_path,
        methods=methods,
        neuron_type=args.neuron_type,
        use_cache=bool(args.use_cache),
        first_trial_only=bool(args.first_trial_only),
        bm_sigma_spike_gap=float(args.bm_sigma_spike_gap),
        pgas_constants=args.pgas_constants,
        pgas_gparam=args.pgas_gparam,
        pgas_output_root=args.pgas_output_root,
        pgas_resample_fs=args.pgas_resample_fs,
        cascade_resample_fs=args.cascade_resample_fs,
        cascade_discretize=bool(not args.cascade_no_discrete),
        ens2_pretrained_root=ens2_pretrained_root,
        cascade_model_root=args.cascade_model_root,
        pgas_maxspikes=args.pgas_maxspikes,
        pgas_fixed_bm_sigma=_parse_optional_float(args.pgas_bm_sigma),
        pgas_c0_first_y=bool(args.pgas_c0_first_y),
        run_tag=args.run_tag,
        trialwise_correlations=bool(args.trialwise_correlations),
    )

    summaries = run_batch(cfg)
    if summaries:
        print(f"Wrote {len(summaries)} summaries.")
        for path in summaries:
            print(f"  {path}")
    else:
        print("No summaries were generated.")


if __name__ == "__main__":
    main()
