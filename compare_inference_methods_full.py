#!/usr/bin/env python3
"""
Batch evaluation script that runs all inference methods across every dataset in
`data/janelia_8f/excitatory` and multiple smoothing targets (raw, 30 Hz, 10 Hz).

Results for each dataset/smoothing combination are cached via the helpers defined in
`compare_inference_methods.py`. The script records summary statistics and discrete spike
times per method so downstream analyses can use the cached outputs without recomputing
inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from c_spikes.inference.eval import build_ground_truth_series, compute_correlations
from c_spikes.inference.pgas import PGAS_RESAMPLE_FS, PgasConfig, pgas_windows_from_result, run_pgas_inference
from c_spikes.inference.smoothing import mean_downsample_trace, resample_trials_to_fs, resolve_smoothing_levels
from c_spikes.inference.types import MethodResult, TrialSeries, ensure_serializable, extract_spike_times, flatten_trials
from c_spikes.inference.ens2 import Ens2Config, run_ens2_inference
from c_spikes.inference.cascade import (
    CASCADE_RESAMPLE_FS,
    CascadeConfig,
    run_cascade_inference,
)
from c_spikes.utils import load_Janelia_data


PGAS_CONSTANTS = Path("parameter_files/constants_GCaMP8_soma.json")
PGAS_GPARAM = Path("src/c_spikes/pgas/20230525_gold.dat")
PGAS_OUTPUT_ROOT = Path("results/pgas_output/comparison")
PGAS_BURNIN = 100
PGAS_NITER = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neuron-type",
        type=str,
        default="Exc",
        help="ENS2 neuron type to evaluate (Exc or Inh).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached inference results when available.",
    )
    parser.add_argument(
        "--skip-ens2",
        action="store_true",
        help="Skip ENS2 inference (useful when only PGAS/CASCADE need reruns).",
    )
    parser.add_argument(
        "--pgas-c0-first-y",
        action="store_true",
        help="Initialize PGAS calcium state C0 to the first observation.",
    )
    parser.add_argument(
        "--bm-sigma-spike-gap",
        type=float,
        default=0.15,
        help="Exclude ±gap seconds around spikes when estimating PGAS bm_sigma (set 0 to disable).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/full_evaluation"),
        help="Directory for summary outputs.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit processing to the first N datasets (for quick spot checks).",
    )
    parser.add_argument(
        "--smoothing-level",
        action="append",
        metavar="LEVEL",
        help="Restrict smoothing levels (options: raw, 30Hz, 10Hz). Repeat to specify multiple.",
    )
    parser.add_argument(
        "--first-trial-only",
        action="store_true",
        help="Restrict processing to the first trial/window per dataset.",
    )
    parser.add_argument(
        "--pgas-resample-fs",
        type=float,
        default=None,
        help=f"Override the PGAS resample frequency (Hz). Default: {PGAS_RESAMPLE_FS}.",
    )
    parser.add_argument(
        "--pgas-maxspikes",
        type=int,
        default=None,
        help="Override the PGAS maxspikes parameter (defaults to heuristic per smoothing rate).",
    )
    parser.add_argument(
        "--pgas-substeps-per-frame",
        type=int,
        default=None,
        help="Force PGAS substeps per observation frame (1 disables sub-stepping).",
    )
    parser.add_argument(
        "--pgas-physics-freq",
        type=float,
        default=None,
        help="Set PGAS internal physics frequency (Hz). Ignored if substeps-per-frame is provided.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional override for the run tag used in output directories; defaults to auto-generated tokens.",
    )
    args = parser.parse_args()

    try:
        smoothing_levels = resolve_smoothing_levels(args.smoothing_level)
    except ValueError as exc:
        parser.error(str(exc))

    if args.pgas_resample_fs is not None and args.pgas_resample_fs <= 0:
        parser.error("--pgas-resample-fs must be positive.")
    if args.pgas_maxspikes is not None and args.pgas_maxspikes <= 0:
        parser.error("--pgas-maxspikes must be positive.")
    pgas_resample_fs = args.pgas_resample_fs or PGAS_RESAMPLE_FS
    run_resample_tag = f"pgas{pgas_resample_fs:g}_cascade{CASCADE_RESAMPLE_FS:g}"
    effective_run_tag = args.run_tag if args.run_tag else run_resample_tag

    data_dir = Path("data/janelia_8f/excitatory")
    dataset_files = sorted(data_dir.glob("*.mat"))
    if not dataset_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")
    if args.max_datasets is not None:
        if args.max_datasets < 1:
            raise ValueError("--max-datasets must be >= 1 when provided.")
        dataset_files = dataset_files[: args.max_datasets]
        print(f"Processing the first {len(dataset_files)} dataset(s) due to --max-datasets.")

    edges_path = Path("results/excitatory_time_stamp_edges.npy")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing timestamp edges file: {edges_path}")
    edges_lookup = np.load(edges_path, allow_pickle=True).item()

    pgas_maxspikes_override = args.pgas_maxspikes

    for dataset_path in dataset_files:
        dataset_tag = dataset_path.stem
        print(f"\nProcessing dataset {dataset_tag}")
        time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))
        trials_native = extract_trials(time_stamps, dff)
        if args.first_trial_only:
            if not trials_native:
                print(f"  Skipping {dataset_tag}: no trials available.")
                continue
            trials_native = trials_native[:1]
        raw_time_flat, raw_trace_flat = flatten_trials(trials_native)
        raw_fs = 1.0 / np.median(np.diff(raw_time_flat))

        if dataset_tag not in edges_lookup:
            print(f"  Skipping {dataset_tag}: no window entry in edges file.")
            continue
        dataset_edges = np.asarray(edges_lookup[dataset_tag], dtype=np.float64)
        if args.first_trial_only:
            dataset_edges = dataset_edges[: len(trials_native)]
            if dataset_edges.size == 0:
                print(f"  Skipping {dataset_tag}: edge file missing entries for first trial.")
                continue

        for label, target in smoothing_levels:
            print(f"\n  Smoothing level: {label}")

            if target is None:
                trials_for_methods = trials_native
                down_time_flat, down_trace_flat = raw_time_flat, raw_trace_flat
                downsample_label = "raw"
            else:
                trials_for_methods = [
                    mean_downsample_trace(trial.times, trial.values, target)
                    for trial in trials_native
                ]
                down_time_flat, down_trace_flat = flatten_trials(trials_for_methods)
                downsample_label = f"{target:.2f}"

            pgas_cfg = PgasConfig(
                dataset_tag=dataset_tag,
                output_root=PGAS_OUTPUT_ROOT,
                constants_file=PGAS_CONSTANTS,
                gparam_file=PGAS_GPARAM,
                resample_fs=pgas_resample_fs,
                niter=PGAS_NITER,
                burnin=PGAS_BURNIN,
                downsample_label=downsample_label,
                maxspikes=pgas_maxspikes_override,
                bm_sigma=None,
                bm_sigma_gap_s=args.bm_sigma_spike_gap,
                edges=dataset_edges,
                use_cache=args.use_cache,
            )

            pgas_result = run_pgas_inference(
                trials_for_methods,
                raw_fs=raw_fs,
                spike_times=spike_times,
                config=pgas_cfg,
            )
            pgas_result.metadata["window_edges"] = dataset_edges.tolist()

            ens2_result: Optional[MethodResult] = None
            if args.skip_ens2:
                print("  [ENS2] skipping inference (--skip-ens2).")
            else:
                ens2_cfg = Ens2Config(
                    dataset_tag=dataset_tag,
                    pretrained_dir=Path("results/Pretrained_models/ens2_published"),
                    neuron_type=args.neuron_type,
                    downsample_label=downsample_label,
                    use_cache=args.use_cache,
                )
                ens2_result = run_ens2_inference(
                    raw_time_stamps=time_stamps,
                    raw_traces=dff,
                    config=ens2_cfg,
                )

            cascade_trials = resample_trials_to_fs(trials_for_methods, CASCADE_RESAMPLE_FS)
            cascade_cfg = CascadeConfig(
                dataset_tag=dataset_tag,
                model_folder=Path("results/Pretrained_models"),
                model_name="Cascade_Universal_30Hz",
                resample_fs=CASCADE_RESAMPLE_FS,
                downsample_label=downsample_label,
                use_cache=args.use_cache,
            )
            cascade_result = run_cascade_inference(
                trials=cascade_trials,
                config=cascade_cfg,
            )

            ref_fs = target if target is not None else raw_fs
            global_start = min(trial.times[0] for trial in trials_native)
            global_end = max(trial.times[-1] for trial in trials_native)
            ref_time, ref_trace = build_ground_truth_series(
                spike_times, global_start, global_end, reference_fs=ref_fs
            )
            pgas_windows = [
                (pgas_result.time_stamps[seg.start], pgas_result.time_stamps[seg.stop - 1])
                for seg in segment_indices(pgas_result.time_stamps, pgas_result.sampling_rate)
                if seg.stop - seg.start > 0
            ]

            method_set = [pgas_result]
            if ens2_result is not None:
                method_set.append(ens2_result)
            method_set.append(cascade_result)

            correlations = compute_correlations(
                method_set,
                ref_time,
                ref_trace,
                windows=pgas_windows,
            )
            print("    Correlations:", correlations)

            summary_dir = args.output_root / effective_run_tag / dataset_tag / label
            summary_dir.mkdir(parents=True, exist_ok=True)

            spike_times_dict = {
                "pgas": extract_spike_times(pgas_result),
                "ens2": extract_spike_times(ens2_result) if ens2_result else None,
                "cascade": extract_spike_times(cascade_result),
            }
            np.savez(
                summary_dir / "discrete_spikes.npz",
                **{k: v if v is not None else np.array([]) for k, v in spike_times_dict.items()},
            )

            summary = {
                "dataset": dataset_tag,
                "smoothing": label,
                "downsample_target": downsample_label,
                "resample_tag": effective_run_tag,
                "correlations": ensure_serializable(correlations),
                "pgas_cache": pgas_result.metadata.get("config", {}),
                "pgas_maxspikes": pgas_result.metadata.get("maxspikes"),
                "pgas_maxspikes_per_bin": pgas_result.metadata.get("maxspikes_per_bin"),
                "pgas_input_resample_fs": pgas_result.metadata.get("input_resample_fs"),
                "ens2_cache": ens2_result.metadata.get("config", {}) if ens2_result else {},
                "cascade_cache": cascade_result.metadata.get("config", {}),
                "cascade_input_resample_fs": cascade_result.metadata.get("input_resample_fs"),
                "pgas_samples": int(np.sum(pgas_result.discrete_spikes))
                if pgas_result.discrete_spikes is not None
                else 0,
                "ens2_samples": int(np.sum(ens2_result.discrete_spikes))
                if (ens2_result and ens2_result.discrete_spikes is not None)
                else 0,
                "cascade_samples": int(np.sum(cascade_result.discrete_spikes))
                if cascade_result.discrete_spikes is not None
                else 0,
                "ens2_skipped": bool(args.skip_ens2),
                "gt_count": int(spike_times.size),
            }
            with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)

            def method_entry(label: str, result: Optional[MethodResult]) -> Optional[Dict[str, object]]:
                if result is None:
                    return None
                meta = result.metadata or {}
                return {
                    "label": label,
                    "method": result.name,
                    "cache_tag": meta.get("cache_tag"),
                    "cache_key": meta.get("cache_key"),
                    "config": ensure_serializable(meta.get("config", {})),
                    "sampling_rate": result.sampling_rate,
                }

            manifest = {
                "run_tag": effective_run_tag,
                "dataset": dataset_tag,
                "smoothing": label,
                "downsample_target": downsample_label,
                "methods": [
                    entry
                    for entry in [
                        method_entry("pgas", pgas_result),
                        method_entry("ens2", ens2_result),
                        method_entry("cascade", cascade_result),
                    ]
                    if entry is not None
                ],
                "artifacts": {
                    "summary": str(summary_dir / "summary.json"),
                    "discrete_spikes": str(summary_dir / "discrete_spikes.npz"),
                },
            }

            with (summary_dir / "comparison.json").open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
