#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from c_spikes.biophys_ml.pipeline import (
    default_cascade_train_config,
    default_synthetic_config,
    generate_synthetic_bundles,
    train_models_for_bundles,
)


def _parse_csv_floats(text: str) -> List[float]:
    return [float(tok.strip()) for tok in text.split(",") if tok.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic datasets from PGAS param_samples and train CASCADE models."
    )
    p.add_argument(
        "--param-samples",
        type=Path,
        action="append",
        required=True,
        help="Path to param_samples_*.dat (repeatable).",
    )
    p.add_argument("--run-root", type=Path, required=True, help="Run root directory.")
    p.add_argument("--run-tag", type=str, required=True, help="Run tag for naming.")
    p.add_argument(
        "--model-root",
        type=Path,
        default=Path("Pretrained_models/BiophysML"),
        help="Output directory for trained CASCADE models.",
    )

    p.add_argument("--burnin", type=int, default=100)
    p.add_argument("--spike-rate-values", type=str, default="6,9,12")
    p.add_argument("--smooth-values", type=str, default="1.3,2.0")
    p.add_argument("--duty-values", type=str, default="0.35,0.45")
    p.add_argument("--noise-fraction", type=float, default=None)
    p.add_argument("--noise-seed-start", type=int, default=0)
    p.add_argument("--noise-seed-stride", type=int, default=1000)
    p.add_argument("--noise-dir", type=Path, default=None)
    p.add_argument("--gparam-path", type=Path, default=Path("src/c_spikes/pgas/20230525_gold.dat"))
    p.add_argument("--force-synth", action="store_true")
    p.add_argument("--no-seed-spikes", action="store_true")

    p.add_argument(
        "--template-model-dir",
        type=Path,
        default=Path("Pretrained_models/CASCADE/Cascade_Universal_30Hz"),
        help="Template CASCADE model directory containing config.yaml.",
    )
    p.add_argument("--model-prefix", type=str, default="bio_ml_cascade")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    synth_cfg = default_synthetic_config()
    synth_cfg.update(
        {
            "burnin": args.burnin,
            "spike_rate_values": _parse_csv_floats(args.spike_rate_values),
            "smooth_values": _parse_csv_floats(args.smooth_values),
            "duty_values": _parse_csv_floats(args.duty_values),
            "noise_fraction": args.noise_fraction,
            "noise_seed_start": args.noise_seed_start,
            "noise_seed_stride": args.noise_seed_stride,
            "noise_dir": None if args.noise_dir is None else str(args.noise_dir),
            "gparam_path": str(args.gparam_path),
            "force_synth": bool(args.force_synth),
            "seed_spikes": bool(not args.no_seed_spikes),
            "synth_tag_prefix": "bio_ml",
        }
    )
    bundles = generate_synthetic_bundles(
        param_samples_paths=args.param_samples,
        run_root=args.run_root,
        run_tag=args.run_tag,
        synthetic_config=synth_cfg,
    )
    train_cfg = default_cascade_train_config()
    train_cfg.update(
        {
            "template_model_dir": str(args.template_model_dir),
            "model_prefix": args.model_prefix,
        }
    )
    records = train_models_for_bundles(
        bundles=bundles,
        run_root=args.run_root,
        model_family="cascade",
        model_root=args.model_root,
        cascade_train_config=train_cfg,
    )
    print(f"Generated {len(bundles)} synthetic bundle(s).")
    print(f"Trained {len(records)} CASCADE model(s).")


if __name__ == "__main__":
    main()

