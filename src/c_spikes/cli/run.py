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
import math
from pathlib import Path
from typing import List, Optional, Sequence

from c_spikes.tensorflow_env import preload_tensorflow_quietly

preload_tensorflow_quietly()

from c_spikes.inference.cache import set_cache_root
from c_spikes.pipeline import RunConfig, run_batch
from c_spikes.inference.pgas import (
    PGAS_BM_SIGMA_DEFAULT,
    PGAS_BM_SIGMA_MAX,
    PGAS_BM_SIGMA_MIN,
    PGAS_NOISE_CALIBRATION_METHOD_DEFAULT,
    PGAS_NOISE_CALIBRATION_METHODS,
    PGAS_NOISE_CALIBRATION_GRANULARITIES,
    PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT,
    PGAS_NOISE_CALIBRATION_SCOPES,
    PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT,
    PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT,
)


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


def _finite_float(value: str) -> float:
    """Argparse converter for finite floating-point values."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _nonnegative_float(value: str) -> float:
    result = _finite_float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_float(value: str) -> float:
    result = _finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_int(value: str) -> int:
    result = _nonnegative_int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/janelia_8f/excitatory"))
    parser.add_argument("--dataset", action="append", metavar="TAG", help="Dataset stem (without .mat). Repeatable.")
    parser.add_argument("--dataset-glob", type=str, default="*.mat", help="Glob under data-root when --dataset is omitted.")
    parser.add_argument("--dataset-list", type=Path, help="File containing dataset stems (one per line).")
    parser.add_argument("--max-datasets", type=int, help="Limit number of datasets processed.")
    parser.add_argument("--smoothing-level", action="append", metavar="LEVEL", help="raw, 30Hz, 10Hz (repeatable).")
    parser.add_argument(
        "--method",
        action="append",
        metavar="NAME",
        help=(
            "Methods to run: pgas, ens2, cascade, oasis. Repeatable. "
            "Default: pgas, ens2, cascade (OASIS is opt-in)."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/full_evaluation"), help="Where to write summaries/manifests.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Inference cache root (defaults to results/inference_cache).",
    )
    parser.add_argument("--edges-path", type=Path, default=Path("results/excitatory_time_stamp_edges.npy"))
    parser.add_argument(
        "--trial-selection-path",
        type=Path,
        help="JSON mapping dataset stem -> trial indices to process (e.g. GUI Batch Selection export).",
    )
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
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Only recompute correlations/summary.json for an existing output directory, using the "
            "cached method outputs referenced by each comparison.json (no inference runs)."
        ),
    )
    parser.add_argument(
        "--first-trial-only",
        action="store_true",
        help=(
            "Restrict each dataset to trial 0, or to the lowest selected trial when "
            "--trial-selection-path is supplied."
        ),
    )
    parser.add_argument("--bm-sigma-spike-gap", type=float, default=0.15, help="Gap around spikes when estimating PGAS bm_sigma.")
    parser.add_argument(
        "--corr-sigma-ms",
        type=float,
        default=50.0,
        help="Gaussian filter width (sigma, ms) used when computing correlations against ground truth.",
    )
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
    parser.add_argument(
        "--pgas-bm-sigma-min",
        type=float,
        default=PGAS_BM_SIGMA_MIN,
        help="Minimum bm_sigma allowed when --pgas-bm-sigma=auto.",
    )
    parser.add_argument(
        "--pgas-bm-sigma-max",
        type=float,
        default=PGAS_BM_SIGMA_MAX,
        help="Maximum bm_sigma allowed when --pgas-bm-sigma=auto.",
    )
    parser.add_argument(
        "--pgas-bm-sigma-use-low-activity-mask",
        action="store_true",
        help="When auto-calibrating bm_sigma, estimate from low-activity regions masked around spikes.",
    )
    parser.add_argument(
        "--pgas-noise-calibration-scope",
        choices=PGAS_NOISE_CALIBRATION_SCOPES,
        default=PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT,
        help=(
            "Data used when auto-calibrating PGAS bm/sigma2. 'inference' uses the "
            "same edge-trimmed windows passed to PGAS; 'full' uses the full selected "
            "epochs/trials while inference and evaluation still use --edges-path."
        ),
    )
    parser.add_argument(
        "--pgas-noise-calibration-granularity",
        choices=PGAS_NOISE_CALIBRATION_GRANULARITIES,
        default=PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT,
        help=(
            "Granularity for auto-calibrating PGAS bm/sigma2. 'dataset' writes one "
            "constants file shared by all selected trials; 'trial' writes per-trial "
            "constants and requires --pgas-bm-sigma=auto."
        ),
    )
    parser.add_argument(
        "--pgas-noise-calibration-method",
        choices=PGAS_NOISE_CALIBRATION_METHODS,
        default=PGAS_NOISE_CALIBRATION_METHOD_DEFAULT,
        help=(
            "Auto-calibration estimator. 'diff' uses robust first/second differences; "
            "'psd' estimates sigma2 from Welch PSD after excluding detected narrowband peaks."
        ),
    )
    parser.add_argument(
        "--pgas-sigma2-target",
        type=str,
        default=None,
        help=(
            "Optional sigma2 mode target used to deterministically set inverse-gamma prior "
            "(beta = target * (alpha + 1)). Use 'none' for default behavior "
            "(calibrated target when bm_sigma is auto; disabled otherwise)."
        ),
    )
    parser.add_argument(
        "--pgas-sigma2-alpha",
        type=str,
        default=None,
        help=(
            "Optional inverse-gamma alpha for sigma2 prior. If omitted and --pgas-sigma2-target "
            "is set, alpha is derived from --pgas-sigma2-prior-strength."
        ),
    )
    parser.add_argument(
        "--pgas-sigma2-prior-strength",
        type=float,
        default=PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT,
        help=(
            "Strength knob used when mapping sigma2 target to IG prior alpha "
            "(alpha = 2 + strength) if --pgas-sigma2-alpha is not set."
        ),
    )
    parser.add_argument(
        "--pgas-keep-output-dat-files",
        action="store_true",
        help=(
            "Keep raw PGAS traj_samples/param_samples/logp .dat files after the cache .mat is "
            "successfully written. By default these large raw dumps are removed; last_params_*.dat "
            "is still preserved."
        ),
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
        "--cascade-model-name",
        type=str,
        default="Cascade_Universal_30Hz",
        help="CASCADE model folder name under --cascade-model-root (default: Cascade_Universal_30Hz).",
    )
    parser.add_argument(
        "--cascade-model-root",
        type=Path,
        default=Path("results/Pretrained_models"),
        help="Root directory containing CASCADE pretrained models.",
    )
    oasis_group = parser.add_argument_group("OASIS options")
    oasis_group.add_argument(
        "--oasis-ar-order",
        type=int,
        choices=(1, 2),
        default=1,
        help="Autoregressive order. Omit --oasis-g to estimate coefficients per trial (default: 1).",
    )
    oasis_group.add_argument(
        "--oasis-g",
        type=_finite_float,
        nargs="+",
        metavar="COEFF",
        help="Per-bin AR coefficient(s): one for AR(1), two for AR(2).",
    )
    oasis_group.add_argument(
        "--oasis-sn",
        type=_nonnegative_float,
        help="Fixed noise standard deviation; omit to estimate it independently per trial.",
    )
    oasis_group.add_argument(
        "--oasis-baseline",
        type=_finite_float,
        help="Fixed fluorescence baseline; omit to optimize it independently per trial.",
    )
    oasis_group.add_argument(
        "--oasis-allow-negative-baseline",
        action="store_true",
        help="Allow an optimized OASIS baseline to be negative.",
    )
    oasis_group.add_argument(
        "--oasis-optimize-g",
        type=_nonnegative_int,
        default=0,
        metavar="EVENTS",
        help="Large isolated events used to optimize AR coefficients; 0 disables it (default: 0).",
    )
    oasis_group.add_argument(
        "--oasis-penalty",
        type=int,
        choices=(0, 1),
        default=1,
        help="Sparsity penalty: 0 for L0 or 1 for L1 (default: 1).",
    )
    oasis_group.add_argument(
        "--oasis-decimate",
        type=_positive_int,
        default=1,
        metavar="FACTOR",
        help="Positive decimation factor used during fitting (default: 1).",
    )
    oasis_group.add_argument(
        "--oasis-max-iter",
        type=_positive_int,
        metavar="COUNT",
        help="Optional positive solver-iteration limit.",
    )
    oasis_group.add_argument(
        "--oasis-shift",
        type=_positive_int,
        metavar="SAMPLES",
        help="Optional positive AR(2) ONNLS block shift.",
    )
    oasis_group.add_argument(
        "--oasis-window",
        type=_positive_int,
        metavar="SAMPLES",
        help="Optional positive AR(2) ONNLS window length.",
    )
    oasis_group.add_argument(
        "--oasis-tol",
        type=_positive_float,
        help="Optional positive AR(2) ONNLS tolerance.",
    )
    oasis_group.add_argument(
        "--oasis-uniformity-rtol",
        type=_nonnegative_float,
        default=5e-3,
        metavar="RTOL",
        help="Relative tolerance for uniform-sampling validation (default: 5e-3).",
    )
    oasis_group.add_argument(
        "--oasis-uniformity-atol",
        type=_nonnegative_float,
        default=1e-9,
        metavar="ATOL",
        help="Absolute tolerance for uniform-sampling validation (default: 1e-9).",
    )
    parser.add_argument(
        "--trialwise-correlations",
        action="store_true",
        help="Also compute and store per-trial correlations in each summary.json.",
    )
    parser.add_argument("--pgas-maxspikes", type=int, help="PGAS maxspikes override.")
    parser.add_argument("--pgas-c0-first-y", action="store_true", help="Initialize PGAS C0 to first observation.")
    parser.add_argument("--run-tag", type=str, help="Optional run-tag override for output directory naming.")
    args = parser.parse_args(argv)
    if args.oasis_g is None:
        args.oasis_g = tuple(None for _ in range(args.oasis_ar_order))
    else:
        args.oasis_g = tuple(float(value) for value in args.oasis_g)
        if len(args.oasis_g) != args.oasis_ar_order:
            parser.error(
                f"--oasis-g requires exactly {args.oasis_ar_order} coefficient(s) "
                f"for AR({args.oasis_ar_order})"
            )
    if args.oasis_ar_order == 1 and any(
        value is not None for value in (args.oasis_shift, args.oasis_window, args.oasis_tol)
    ):
        parser.error("--oasis-shift, --oasis-window, and --oasis-tol require --oasis-ar-order 2")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.cache_root is not None:
        set_cache_root(args.cache_root)

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
        trial_selection_path=args.trial_selection_path,
        methods=methods,
        neuron_type=args.neuron_type,
        use_cache=bool(args.use_cache),
        eval_only=bool(args.eval_only),
        first_trial_only=bool(args.first_trial_only),
        bm_sigma_spike_gap=float(args.bm_sigma_spike_gap),
        corr_sigma_ms=float(args.corr_sigma_ms),
        pgas_constants=args.pgas_constants,
        pgas_gparam=args.pgas_gparam,
        pgas_output_root=args.pgas_output_root,
        pgas_resample_fs=args.pgas_resample_fs,
        cascade_resample_fs=args.cascade_resample_fs,
        cascade_discretize=bool(not args.cascade_no_discrete),
        ens2_pretrained_root=ens2_pretrained_root,
        cascade_model_root=args.cascade_model_root,
        cascade_model_name=str(args.cascade_model_name),
        oasis_g=args.oasis_g,
        oasis_sn=args.oasis_sn,
        oasis_b=args.oasis_baseline,
        oasis_b_nonneg=bool(not args.oasis_allow_negative_baseline),
        oasis_optimize_g=int(args.oasis_optimize_g),
        oasis_penalty=int(args.oasis_penalty),
        oasis_decimate=int(args.oasis_decimate),
        oasis_max_iter=args.oasis_max_iter,
        oasis_shift=args.oasis_shift,
        oasis_window=args.oasis_window,
        oasis_tol=args.oasis_tol,
        oasis_uniformity_rtol=float(args.oasis_uniformity_rtol),
        oasis_uniformity_atol=float(args.oasis_uniformity_atol),
        pgas_maxspikes=args.pgas_maxspikes,
        pgas_fixed_bm_sigma=_parse_optional_float(args.pgas_bm_sigma),
        pgas_bm_sigma_min=float(args.pgas_bm_sigma_min),
        pgas_bm_sigma_max=float(args.pgas_bm_sigma_max),
        pgas_keep_output_dat_files=bool(args.pgas_keep_output_dat_files),
        pgas_bm_sigma_use_low_activity_mask=bool(args.pgas_bm_sigma_use_low_activity_mask),
        pgas_sigma2_target=_parse_optional_float(args.pgas_sigma2_target),
        pgas_sigma2_alpha=_parse_optional_float(args.pgas_sigma2_alpha),
        pgas_sigma2_prior_strength=float(args.pgas_sigma2_prior_strength),
        pgas_noise_calibration_scope=str(args.pgas_noise_calibration_scope),
        pgas_noise_calibration_granularity=str(args.pgas_noise_calibration_granularity),
        pgas_noise_calibration_method=str(args.pgas_noise_calibration_method),
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
