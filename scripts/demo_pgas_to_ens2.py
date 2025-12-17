#!/usr/bin/env python3
"""
End-to-end demo:
  1) Take one or more PGAS param_samples files.
  2) Generate synthetic ground-truth datasets via syn_gen (with manifest logging).
  3) Train a custom ENS2 model on the combined synthetic datasets (with manifest logging).
  4) Optionally run demo_compare_methods.py against the trained ENS2.

Usage example:
  python scripts/demo_pgas_to_ens2.py \
    --param-samples results/pgas_output/test_refactor/param_samples_jGCaMP8f_ANM471993_cell01_pgas_new_sraw_ms3_rs120_bm0p05_trial0.dat \
    --model-name ens2_synth_cell01 \
    --run-tag trial01 \
    --train-ens2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from c_spikes.syn_gen import build_synthetic_ground_truth_batch
from c_spikes.ens2 import train_model
from c_spikes.syn_gen.noise_preprocess import prepare_noise_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--param-samples",
        type=Path,
        action="append",
        required=True,
        help="Path to a PGAS param_samples_*.dat file (repeatable).",
    )
    p.add_argument("--burnin", type=int, default=100, help="Burn-in rows to discard from each param_samples file.")
    p.add_argument("--spike-rate", type=float, default=2.0, help="Nominal spike rate for syn_gen.")
    p.add_argument(
        "--spike-params",
        type=float,
        nargs=2,
        default=(5.0, 0.5),
        metavar=("SMOOTH_SEC", "R_NONZERO"),
        help="syn_gen spike_params (smoothtime, rnonzero).",
    )
    p.add_argument(
        "--noise-dir",
        type=Path,
        default=None,
        help="Noise directory for syn_gen (default: syn_gen/gt_noise_dir).",
    )
    p.add_argument(
        "--noise-target-fs",
        type=float,
        default=None,
        help=(
            "If set, preprocess the syn_gen noise directory to this sampling rate (Hz) before synthesis. "
            "Writes a cached copy under results/inference_cache/noise_downsample/ and uses it for syn_gen."
        ),
    )
    p.add_argument(
        "--gparam-path",
        type=Path,
        default=Path("src/c_spikes/pgas/20230525_gold.dat"),
        help="GCaMP parameter file used by syn_gen (sensor-specific).",
    )
    p.add_argument(
        "--noise-fraction",
        type=float,
        default=1.0,
        help="Fraction of noise files to use from noise_dir (1.0=all).",
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        action="append",
        default=None,
        help="Optional seed(s) for noise file subsampling (repeat to concatenate subsets).",
    )
    p.add_argument(
        "--noise-seed-base",
        type=int,
        default=None,
        help=(
            "Optional base seed for multi-param_samples runs. If set, each param_samples file "
            "uses a unique noise seed (base + index). Mutually exclusive with --noise-seed."
        ),
    )
    p.add_argument(
        "--synth-tag-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to each synthetic dataset tag to avoid reusing synth_* dirs across sweeps.",
    )
    p.add_argument("--model-name", type=str, required=True, help="Name for the custom ENS2 model folder.")
    p.add_argument(
        "--model-root",
        type=Path,
        default=Path("results/Pretrained_models"),
        help="Root under which to place the custom ENS2 model.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path (default: <model_root>/<model_name>/ens2_manifest.json).",
    )
    p.add_argument("--run-tag", type=str, default=None, help="Run tag to log into manifest entries.")
    p.add_argument("--train-ens2", action="store_true", help="If set, train ENS2 after generating synthetic data.")
    p.add_argument("--sampling-rate", type=float, default=60.0, help="ENS2 sampling rate (Hz).")
    p.add_argument("--smoothing-std", type=float, default=0.025, help="ENS2 smoothing std (seconds).")
    p.add_argument(
        "--neuron-type",
        type=str,
        default="Exc",
        choices=["Exc", "exc", "Inh", "inh"],
        help="ENS2 neuron type for checkpoint naming.",
    )
    p.add_argument(
        "--run-compare",
        action="store_true",
        help="If set, run demo_compare_methods.py against the trained ENS2 checkpoint.",
    )
    p.add_argument(
        "--stock-ens2-root",
        type=Path,
        default=Path("results/Pretrained_models/ens2_published"),
        help="Root for the stock/published ENS2 checkpoints used as baseline.",
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        help="Include PGAS and CASCADE in demo_compare_methods (default: only ENS2).",
    )
    p.add_argument(
        "--compare-use-cache",
        action="store_true",
        help="Allow demo_compare_methods to reuse cached results (default: force recompute).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset (.mat) to use for demo_compare_methods when --run-compare is set.",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="Optional smoothing Hz for demo_compare_methods (None=raw).",
    )
    p.add_argument(
        "--edges-file",
        type=Path,
        default=None,
        help="Optional edges file for demo_compare_methods.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.noise_seed_base is not None and args.noise_seed is not None:
        raise ValueError("Provide only one of --noise-seed or --noise-seed-base.")

    resolved_noise_dir: Path | None = None
    if args.noise_target_fs is not None:
        import c_spikes.syn_gen as syn_gen_pkg

        syn_gen_dir = Path(syn_gen_pkg.__file__).resolve().parent
        if args.noise_dir is None:
            resolved_noise_dir = syn_gen_dir / "gt_noise_dir"
        else:
            resolved_noise_dir = args.noise_dir
            if not resolved_noise_dir.is_absolute():
                resolved_noise_dir = syn_gen_dir / resolved_noise_dir
        resolved_noise_dir = prepare_noise_dir(resolved_noise_dir, target_fs=float(args.noise_target_fs))

    manifest_path = (
        args.manifest
        if args.manifest is not None
        else args.model_root / args.model_name / "ens2_manifest.json"
    )

    # Build param specs for batch generation
    param_specs: List[dict] = []
    for idx, ps_path in enumerate(args.param_samples):
        base_tag = Path(ps_path).stem.replace("param_samples_", "")
        tag = base_tag if args.synth_tag_suffix is None else f"{base_tag}__{args.synth_tag_suffix}"
        noise_seed = (
            int(args.noise_seed_base) + idx
            if args.noise_seed_base is not None
            else args.noise_seed
        )
        param_specs.append(
            {
                "param_samples_path": ps_path,
                "burnin": args.burnin,
                "spike_rate": args.spike_rate,
                "spike_params": args.spike_params,
                "noise_dir": resolved_noise_dir if resolved_noise_dir is not None else args.noise_dir,
                "noise_fraction": args.noise_fraction,
                "noise_seed": noise_seed,
                "tag": tag,
                "run_tag": args.run_tag,
            }
        )

    # 1-2) Generate synthetic datasets and log to manifest
    cparams_map = build_synthetic_ground_truth_batch(
        param_specs,
        gparam_path=args.gparam_path,
        output_root=Path("results"),
        manifest_path=manifest_path,
        manifest_model_name=args.model_name,
    )

    synth_dirs = [
        Path("results") / "Ground_truth" / f"synth_{spec['tag']}"
        for spec in param_specs
    ]

    # 3) Optionally train ENS2 on the combined synthetic datasets
    if args.train_ens2:
        checkpoint_path = train_model(
            model_name=args.model_name,
            synth_gt_dir=synth_dirs,
            model_root=args.model_root,
            neuron_type=args.neuron_type,
            sampling_rate=args.sampling_rate,
            smoothing_std=args.smoothing_std,
            manifest_path=manifest_path,
            run_tag=args.run_tag,
        )
        print(f"[ens2] Trained checkpoint: {checkpoint_path}")
    else:
        print("[ens2] Skipped training (--train-ens2 not set).")

    # 4) Optionally run demo_compare_methods.py using the trained ENS2
    if args.run_compare:
        if args.dataset is None:
            raise ValueError("--dataset is required when --run-compare is set.")
        ens2_root = args.model_root / args.model_name
        cmd = [
            "python",
            "scripts/demo_compare_methods.py",
            "--dataset",
            str(args.dataset),
            "--ens2-pretrained-root",
            str(args.stock_ens2_root),
            "--ens2-custom-root",
            str(ens2_root),
        ]
        if not args.compare_use_cache:
            cmd.append("--no-cache")
        if not args.compare_all:
            cmd.extend(["--skip-pgas", "--skip-cascade"])
        if args.smoothing is not None:
            cmd.extend(["--smoothing", str(args.smoothing)])
        if args.edges_file is not None:
            cmd.extend(["--edges-file", str(args.edges_file)])
        print("[compare] Running:", " ".join(cmd))
        import subprocess

        subprocess.run(cmd, check=True)

    # Print summary
    print("Generated synthetic datasets:")
    for tag, cp in cparams_map.items():
        print(f"  tag={tag} cparams_mean={cp}")


if __name__ == "__main__":
    main()
