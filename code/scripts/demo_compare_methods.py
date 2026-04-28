#!/usr/bin/env python3
"""
Demo: run PGAS, ENS2, and CASCADE on a user-specified dataset and compare outputs.

Features:
  - Optional smoothing/downsampling (set a target Hz or use native rate).
  - Optional PGAS/CASCADE resample overrides.
  - Optional trimming via edges file or start/end times.
  - Prints correlations and shows overlay plots (spike_prob + discrete spikes).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from c_spikes.inference.workflow import (
    DatasetRunConfig,
    MethodSelection,
    SmoothingLevel,
    run_inference_for_dataset,
)
from c_spikes.inference.pgas import PGAS_BM_SIGMA_DEFAULT
from c_spikes.utils import load_Janelia_data


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"none", "null", "auto", "estimate", "estimated"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
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
    ax.set_ylabel("Signal / spike prob (offset per method)")
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
    selection = MethodSelection(
        run_pgas=not args.skip_pgas,
        run_ens2=not args.skip_ens2,
        run_cascade=not args.skip_cascade,
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
        cascade_discretize=bool(not args.cascade_no_discrete),
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
