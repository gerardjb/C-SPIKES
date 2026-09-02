#!/usr/bin/env python3
"""
Demo: run PGAS, ENS2, CASCADE, and optionally OASIS on a dataset and compare outputs.

Features:
  - Optional smoothing/downsampling (set a target Hz or use native rate).
  - Optional PGAS/CASCADE resample overrides.
  - Opt-in OASIS inference with per-trial AR(1) or AR(2) deconvolution.
  - Optional trimming via edges file or start/end times.
  - Prints correlations and shows overlay plots (spike_prob + discrete spikes).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from c_spikes.inference.workflow import (
    DatasetRunConfig,
    MethodSelection,
    SmoothingLevel,
    run_inference_for_dataset,
)
from c_spikes.inference.pgas import (
    PGAS_BM_SIGMA_DEFAULT,
    PGAS_BM_SIGMA_MAX,
    PGAS_BM_SIGMA_MIN,
    PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT,
)
from c_spikes.utils import load_Janelia_data


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"none", "null", "auto", "estimate", "estimated"}:
        return None
    return float(value)


def _resolve_oasis_g(
    ar_order: int,
    coefficients: Optional[Sequence[float]],
) -> tuple[Optional[float], ...]:
    if coefficients is None:
        return tuple(None for _ in range(ar_order))
    resolved = tuple(float(value) for value in coefficients)
    if len(resolved) != ar_order:
        raise ValueError(
            f"--oasis-g requires exactly {ar_order} coefficient(s) for AR({ar_order})."
        )
    return resolved


def _validate_oasis_args(args: argparse.Namespace) -> None:
    if args.oasis_optimize_g < 0:
        raise ValueError("--oasis-optimize-g must be non-negative.")
    for name in ("oasis_decimate", "oasis_max_iter", "oasis_shift", "oasis_window"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.oasis_sn is not None and args.oasis_sn < 0:
        raise ValueError("--oasis-sn must be non-negative.")
    if args.oasis_tol is not None and args.oasis_tol <= 0:
        raise ValueError("--oasis-tol must be positive.")
    if args.oasis_event_threshold is not None and (
        not np.isfinite(args.oasis_event_threshold) or args.oasis_event_threshold <= 0
    ):
        raise ValueError("--oasis-event-threshold must be positive and finite.")
    if args.oasis_discrete_mode == "support" and args.oasis_event_threshold is None:
        raise ValueError("--oasis-event-threshold is required for support mode.")
    if args.oasis_discrete_mode == "none" and args.oasis_event_threshold is not None:
        raise ValueError("--oasis-event-threshold requires support mode.")
    if args.oasis_discrete_mode == "none" and args.oasis_threshold_units != "absolute":
        raise ValueError("--oasis-threshold-units requires support mode.")
    if args.oasis_ar_order == 1 and any(
        value is not None for value in (args.oasis_shift, args.oasis_window, args.oasis_tol)
    ):
        raise ValueError(
            "--oasis-shift, --oasis-window, and --oasis-tol require --oasis-ar-order 2."
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to .mat file with time_stamps and dff.")
    parser.add_argument("--smoothing", type=float, default=None, help="Target Hz for pre-inference smoothing (None=raw).")
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
        default=Path("results/pgas_output/demo"),
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
        "--pgas-sigma2-target",
        type=str,
        default=None,
        help=(
            "Optional sigma2 mode target used to set the inverse-gamma prior "
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
    parser.add_argument("--pgas-resample", type=float, default=None, help="PGAS resample Hz (None=use native).")
    parser.add_argument(
        "--cascade-resample",
        type=float,
        default=None,
        help="CASCADE input resample Hz (default: None => use input sampling rate).",
    )
    parser.add_argument(
        "--cascade-no-discrete",
        action="store_true",
        help="Skip CASCADE discrete-spike inference (avoids slow/hanging discretization; correlations still computed from spike_prob).",
    )
    parser.add_argument(
        "--trialwise-correlations",
        action="store_true",
        help="Also compute per-trial correlations (printed in JSON summary output object).",
    )
    parser.add_argument("--edges-file", type=Path, help="Optional edges npy (dict dataset->edges) for trimming.")
    parser.add_argument("--start-time", type=float, help="Manual trim start (sec).")
    parser.add_argument("--end-time", type=float, help="Manual trim end (sec).")
    parser.add_argument("--epoch-start", type=int, default=None, help="Start trial/epoch index (0-based).")
    parser.add_argument("--epoch-stop", type=int, default=None, help="Stop trial/epoch index (exclusive).")
    parser.add_argument("--no-cache", action="store_true", help="Disable all method caches (force recompute).")
    parser.add_argument("--skip-pgas", action="store_true", help="Skip PGAS.")
    parser.add_argument("--skip-ens2", action="store_true", help="Skip ENS2.")
    parser.add_argument("--skip-cascade", action="store_true", help="Skip CASCADE.")
    parser.add_argument(
        "--run-oasis",
        action="store_true",
        help="Run OASIS in addition to the default comparison methods.",
    )
    parser.add_argument(
        "--oasis-ar-order",
        type=int,
        choices=(1, 2),
        default=1,
        help="OASIS autoregressive order (default: 1).",
    )
    parser.add_argument(
        "--oasis-g",
        type=float,
        nargs="+",
        metavar="COEFF",
        help="Per-bin AR coefficient(s); omit to estimate independently per trial.",
    )
    parser.add_argument("--oasis-sn", type=float, help="Fixed OASIS noise level; omit to estimate.")
    parser.add_argument(
        "--oasis-baseline",
        type=float,
        help="Fixed fluorescence baseline; omit to estimate.",
    )
    parser.add_argument(
        "--oasis-allow-negative-baseline",
        action="store_true",
        help="Allow an estimated OASIS baseline to be negative.",
    )
    parser.add_argument(
        "--oasis-optimize-g",
        type=int,
        default=0,
        metavar="EVENTS",
        help="Large isolated events used to optimize AR coefficients; 0 disables it.",
    )
    parser.add_argument("--oasis-penalty", type=int, choices=(0, 1), default=1)
    parser.add_argument("--oasis-decimate", type=int, default=1)
    parser.add_argument("--oasis-max-iter", type=int)
    parser.add_argument("--oasis-shift", type=int, help="AR(2)-only ONNLS shift.")
    parser.add_argument("--oasis-window", type=int, help="AR(2)-only ONNLS window.")
    parser.add_argument("--oasis-tol", type=float, help="AR(2)-only ONNLS tolerance.")
    parser.add_argument(
        "--oasis-discrete-mode",
        choices=("none", "support"),
        default="none",
        help=(
            "Optional binary event-support output; 'none' preserves continuous-only "
            "OASIS output (default: none)."
        ),
    )
    parser.add_argument(
        "--oasis-event-threshold",
        type=float,
        metavar="VALUE",
        help=(
            "Positive threshold used in support mode. It is an event-support cutoff, "
            "not a calibrated spike count."
        ),
    )
    parser.add_argument(
        "--oasis-threshold-units",
        choices=("absolute", "noise_scaled"),
        default="absolute",
        help=(
            "Interpret the event threshold as an absolute OASIS amplitude or a "
            "per-trial noise-scaled factor (default: absolute)."
        ),
    )
    parser.add_argument(
        "--ens2-pretrained-root",
        type=Path,
        default=Path("results/Pretrained_models/ens2_published"),
        help="Root directory for the stock/published ENS2 checkpoints.",
    )
    parser.add_argument(
        "--ens2-custom-root",
        type=Path,
        default=None,
        help="Optional custom ENS2 checkpoint root (runs an additional ENS2 labeled 'ens2_custom').",
    )
    parser.add_argument(
        "--corr-sigma-ms",
        type=float,
        default=50.0,
        help="Gaussian sigma (ms) used to smooth GT spikes and method predictions for correlation (default: 50).",
    )
    parser.add_argument("--plot", action="store_true", help="Show overlay plots.")
    return parser.parse_args(argv)


def plot_overlay(
    raw_time: np.ndarray,
    raw_trace: np.ndarray,
    spike_times: np.ndarray,
    methods: Mapping[str, object],
    title: str,
    xlim: Optional[tuple[float, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw_time, raw_trace, color="k", linewidth=0.7, alpha=0.9, label="Raw dff")
    if spike_times.size:
        ax.vlines(
            spike_times,
            ymin=np.nanmin(raw_trace),
            ymax=np.nanmax(raw_trace),
            color="tab:red",
            alpha=0.2,
            linewidth=0.8,
            label="GT spikes",
        )
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    for idx, (label, m) in enumerate((methods or {}).items()):
        c = colors[idx % len(colors)]
        times = np.asarray(getattr(m, "time_stamps"), dtype=float)
        values = np.asarray(getattr(m, "spike_prob"), dtype=float) - (idx + 1) * 1
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            continue
        valid_times = times[finite_mask]
        valid_vals = values[finite_mask]
        ax.plot(valid_times, valid_vals, label=f"{label} spike_prob", color=c, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal / continuous inference output (offset per method)")
    ax.legend()
    if xlim:
        ax.set_xlim(*xlim)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    time_stamps, dff, spike_times = load_Janelia_data(str(args.dataset))
    dataset_tag = args.dataset.stem

    total_trials = time_stamps.shape[0]
    epoch_start = args.epoch_start if args.epoch_start is not None else 0
    epoch_stop = args.epoch_stop if args.epoch_stop is not None else total_trials
    if epoch_start < 0 or epoch_stop > total_trials or epoch_start >= epoch_stop:
        raise ValueError(f"Invalid epoch range [{epoch_start}, {epoch_stop}) for {total_trials} trials.")

    # Optional epoch slicing
    if epoch_start != 0 or epoch_stop != total_trials:
        time_stamps = time_stamps[epoch_start:epoch_stop]
        dff = dff[epoch_start:epoch_stop]

    edges = None
    if (args.start_time is not None) ^ (args.end_time is not None):
        raise ValueError("Provide both --start-time and --end-time, or neither.")
    if args.start_time is not None and args.end_time is not None:
        if args.end_time <= args.start_time:
            raise ValueError("end-time must exceed start-time.")
        edges = np.array([[args.start_time, args.end_time]] * time_stamps.shape[0], dtype=float)
    elif args.edges_file and args.edges_file.exists():
        edges_lookup = np.load(args.edges_file, allow_pickle=True).item()
        if dataset_tag in edges_lookup:
            candidate = np.asarray(edges_lookup[dataset_tag], dtype=float)
            if candidate.shape[0] >= epoch_stop:
                edges = candidate[epoch_start:epoch_stop]
            else:
                print(
                    f"[WARN] Edges for dataset '{dataset_tag}' shorter than requested epoch slice; skipping trim."
                )
        else:
            print(f"[WARN] Dataset '{dataset_tag}' not in edges file {args.edges_file}; skipping trim.")

    trial_bounds = np.column_stack((time_stamps[:, 0], time_stamps[:, -1]))
    if edges is not None and edges.shape[0] != time_stamps.shape[0]:
        print(f"[WARN] Edges shape {edges.shape} does not match selected trials; ignoring edges.")
        edges = None
    if edges is not None:
        clipped = []
        for idx, (start, end) in enumerate(edges):
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                raise ValueError(f"Invalid edge bounds ({start}, {end}) for trial {idx}.")
            s = max(start, trial_bounds[idx, 0])
            e = min(end, trial_bounds[idx, 1])
            if e <= s:
                raise ValueError(
                    f"Edge window ({start}, {end}) for trial {idx} is outside data range "
                    f"[{trial_bounds[idx,0]}, {trial_bounds[idx,1]}]."
                )
            clipped.append((s, e))
        edges = np.asarray(clipped, dtype=float)

    windows_for_spikes = edges if edges is not None else trial_bounds
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    if spike_times.size:
        mask = np.zeros(spike_times.shape, dtype=bool)
        for start, end in windows_for_spikes:
            mask |= (spike_times >= start) & (spike_times <= end)
        spike_times = spike_times[mask]

    smoothing = SmoothingLevel(target_fs=args.smoothing)
    oasis_g = _resolve_oasis_g(args.oasis_ar_order, args.oasis_g)
    _validate_oasis_args(args)
    selection = MethodSelection(
        run_pgas=not args.skip_pgas,
        run_ens2=not args.skip_ens2,
        run_cascade=not args.skip_cascade,
        run_oasis=bool(args.run_oasis),
    )
    cfg = DatasetRunConfig(
        dataset_path=args.dataset,
        smoothing=smoothing,
        reference_fs=None,
        edges=edges,
        selection=selection,
        use_cache=not args.no_cache,
        bm_sigma_gap_s=0.15,
        corr_sigma_ms=float(args.corr_sigma_ms),
        pgas_resample_fs=args.pgas_resample,
        cascade_resample_fs=args.cascade_resample,
        pgas_fixed_bm_sigma=_parse_optional_float(args.pgas_bm_sigma),
        pgas_bm_sigma_min=float(args.pgas_bm_sigma_min),
        pgas_bm_sigma_max=float(args.pgas_bm_sigma_max),
        pgas_bm_sigma_use_low_activity_mask=bool(args.pgas_bm_sigma_use_low_activity_mask),
        pgas_sigma2_target=_parse_optional_float(args.pgas_sigma2_target),
        pgas_sigma2_alpha=_parse_optional_float(args.pgas_sigma2_alpha),
        pgas_sigma2_prior_strength=float(args.pgas_sigma2_prior_strength),
        cascade_discretize=bool(not args.cascade_no_discrete),
        oasis_g=oasis_g,
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
        oasis_discrete_mode=str(args.oasis_discrete_mode),
        oasis_event_threshold=args.oasis_event_threshold,
        oasis_threshold_units=str(args.oasis_threshold_units),
        trialwise_correlations=bool(args.trialwise_correlations),
    )

    outputs = run_inference_for_dataset(
        cfg,
        pgas_constants=args.pgas_constants,
        pgas_gparam=args.pgas_gparam,
        pgas_output_root=args.pgas_output_root,
        ens2_pretrained_root=args.ens2_pretrained_root,
        cascade_model_root=Path("results/Pretrained_models"),
        dataset_data=(time_stamps, dff, spike_times),
    )

    methods = outputs["methods"]
    correlations = outputs.get("correlations", {})

    # Optional second ENS2 run with custom checkpoints
    if args.ens2_custom_root is not None and not args.skip_ens2:
        custom_cfg = DatasetRunConfig(
            dataset_path=args.dataset,
            smoothing=smoothing,
            reference_fs=None,
            edges=edges,
            selection=MethodSelection(run_pgas=False, run_ens2=True, run_cascade=False),
            use_cache=not args.no_cache,
            bm_sigma_gap_s=0.15,
            corr_sigma_ms=float(args.corr_sigma_ms),
            pgas_resample_fs=args.pgas_resample,
            cascade_resample_fs=args.cascade_resample,
            pgas_fixed_bm_sigma=_parse_optional_float(args.pgas_bm_sigma),
            pgas_bm_sigma_min=float(args.pgas_bm_sigma_min),
            pgas_bm_sigma_max=float(args.pgas_bm_sigma_max),
            pgas_bm_sigma_use_low_activity_mask=bool(args.pgas_bm_sigma_use_low_activity_mask),
            pgas_sigma2_target=_parse_optional_float(args.pgas_sigma2_target),
            pgas_sigma2_alpha=_parse_optional_float(args.pgas_sigma2_alpha),
            pgas_sigma2_prior_strength=float(args.pgas_sigma2_prior_strength),
            cascade_discretize=bool(not args.cascade_no_discrete),
        )
        custom_outputs = run_inference_for_dataset(
            custom_cfg,
            pgas_constants=args.pgas_constants,
            pgas_gparam=args.pgas_gparam,
            pgas_output_root=args.pgas_output_root,
            ens2_pretrained_root=args.ens2_custom_root,
            cascade_model_root=Path("results/Pretrained_models"),
            dataset_data=(time_stamps, dff, spike_times),
        )
        if "ens2" in custom_outputs["methods"]:
            methods["ens2_custom"] = custom_outputs["methods"]["ens2"]
            if "ens2" in custom_outputs.get("correlations", {}):
                correlations["ens2_custom"] = custom_outputs["correlations"]["ens2"]
    print("Methods run:", list(methods.keys()))

    # Correlations (if spike_times provided)
    if spike_times.size > 0 and methods:
        print("Correlations vs GT:", correlations)
    else:
        print("No spike_times provided; skipping correlation.")

    if args.plot:
        plot_overlay(
            outputs["raw_time"],
            outputs["raw_trace"],
            spike_times,
            methods,
            title=f"{dataset_tag}: raw + GT spikes + methods",
        )


if __name__ == "__main__":
    main()
