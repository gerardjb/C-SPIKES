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

from compare_inference_methods import (
    MethodResult,
    build_ground_truth_series,
    compute_correlations,
    derive_bm_sigma,
    extract_trials,
    hash_array,
    hash_series,
    load_Janelia_data,
    load_method_cache,
    load_pgas_method_result,
    mean_downsample_trace,
    run_cascade_inference,
    run_ens2_inference,
    run_pgas_inference,
    refine_pgas_result,
    save_method_cache,
    trim_trials_by_edges,
    flatten_trials,
    ensure_serializable,
    segment_indices,
    prepare_constants_with_params,
    maxspikes_for_rate,
    resample_trials_to_fs,
    PGAS_RESAMPLE_FS,
    CASCADE_RESAMPLE_FS,
    format_tag_token,
    PGAS_MAX_SPIKES_PER_BIN,
    build_low_activity_mask,
)

SMOOTHING_LEVELS: Sequence[Tuple[str, Optional[float]]] = [
    ("raw", None),
    ("30Hz", 30.0),
    ("10Hz", 10.0),
]

CASCADE_MODEL = "Cascade_Universal_30Hz"
CASCADE_MODEL_FOLDER = Path("results/Pretrained_models")
ENS2_PRETRAINED = Path("results/Pretrained_models/ens2_published")
PGAS_CONSTANTS = Path("parameter_files/constants_GCaMP8_soma.json")
PGAS_GPARAM = Path("src/c_spikes/pgas/20230525_gold.dat")
PGAS_OUTPUT_ROOT = Path("results/pgas_output/comparison")
PGAS_BURNIN = 100
PGAS_NITER = 200
CASCADE_RESAMPLE_TOKEN = format_tag_token(f"{CASCADE_RESAMPLE_FS:g}")


def resolve_smoothing_levels(
    selection: Optional[Sequence[str]],
) -> Sequence[Tuple[str, Optional[float]]]:
    """Map user-provided smoothing tokens to (label, target_fs) pairs."""
    if not selection:
        return SMOOTHING_LEVELS
    mapping: Dict[str, Tuple[str, Optional[float]]] = {
        "raw": ("raw", None),
        "30": ("30Hz", 30.0),
        "30hz": ("30Hz", 30.0),
        "10": ("10Hz", 10.0),
        "10hz": ("10Hz", 10.0),
    }
    resolved: List[Tuple[str, Optional[float]]] = []
    for token in selection:
        key = token.strip().lower()
        if key not in mapping:
            raise ValueError(
                f"Invalid smoothing level '{token}'. Expected one of: "
                + ", ".join(sorted(mapping.keys()))
            )
        entry = mapping[key]
        if entry not in resolved:
            resolved.append(entry)
    return resolved


def extract_spike_times(result: MethodResult) -> Optional[np.ndarray]:
    if result.discrete_spikes is None:
        return None
    counts = np.asarray(result.discrete_spikes, dtype=float)
    times = []
    for t, c in zip(result.time_stamps, counts):
        n = int(round(c))
        if n > 0:
            times.extend([t] * n)
    return np.asarray(times, dtype=float)


def fetch_pgas_result(
    dataset_tag: str,
    trials_for_pgas,
    dataset_edges: np.ndarray,
    downsample_label: str,
    downsample_target: Optional[float],
    raw_fs: float,
    use_cache: bool,
    refine_bins: bool = False,
    pgas_resample_fs: float = PGAS_RESAMPLE_FS,
    maxspikes_override: Optional[int] = None,
    bm_sigma_series: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    substeps_per_frame: Optional[int] = None,
    physics_frequency_hz: Optional[float] = None,
    c0_is_first_y: Optional[bool] = None,
) -> MethodResult:
    trials_for_pgas = list(trials_for_pgas)
    trials_resampled = resample_trials_to_fs(trials_for_pgas, pgas_resample_fs)
    time_flat, trace_flat = flatten_trials(trials_resampled)
    trace_hash = hash_series(time_flat, trace_flat)
    maxspikes = maxspikes_override if maxspikes_override is not None else maxspikes_for_rate(
        downsample_target, raw_fs
    )
    if bm_sigma_series is not None and bm_sigma_series[0].size >= 2:
        sigma_times, sigma_values = bm_sigma_series
    else:
        sigma_times, sigma_values = time_flat, trace_flat
    bm_sigma = derive_bm_sigma(
        sigma_times,
        sigma_values,
        target_fs=pgas_resample_fs,
    )
    bm_token = format_tag_token(f"{bm_sigma:.3g}")
    constants_path = prepare_constants_with_params(
        PGAS_CONSTANTS,
        maxspikes=maxspikes,
        bm_sigma=bm_sigma,
        substeps_per_frame=substeps_per_frame,
        physics_frequency_hz=physics_frequency_hz,
        c0_is_first_y=c0_is_first_y,
    )
    label_token = format_tag_token(downsample_label)
    pgas_resample_token = format_tag_token(f"{pgas_resample_fs:g}")
    substep_token = None
    if substeps_per_frame is not None:
        substep_token = f"spf{substeps_per_frame}"
    elif physics_frequency_hz is not None:
        substep_token = f"pfreq{format_tag_token(f'{physics_frequency_hz:g}')}"
    pgas_run_tag = f"{dataset_tag}_s{label_token}_ms{maxspikes}_rs{pgas_resample_token}_bm{bm_token}"
    if substep_token:
        pgas_run_tag = f"{pgas_run_tag}_{substep_token}"
    config = {
        "niter": PGAS_NITER,
        "burnin": PGAS_BURNIN,
        "downsample_target": downsample_label,
        "constants_file": str(constants_path),
        "gparam_file": str(PGAS_GPARAM),
        "edge_hash": hash_array(dataset_edges),
        "maxspikes": maxspikes,
        "input_resample_fs": pgas_resample_fs,
        "adaptive_refine": refine_bins,
        "bm_sigma": bm_sigma,
    }
    if substeps_per_frame is not None:
        config["substeps_per_frame"] = substeps_per_frame
    if physics_frequency_hz is not None:
        config["physics_frequency_hz"] = physics_frequency_hz
    if c0_is_first_y is not None:
        config["c0_is_first_y"] = bool(c0_is_first_y)
    result: Optional[MethodResult] = None
    if use_cache:
        result = load_method_cache("pgas", pgas_run_tag, config, trace_hash)
        if result:
            result.metadata.setdefault("input_resample_fs", pgas_resample_fs)
            result.metadata.setdefault("maxspikes_per_bin", PGAS_MAX_SPIKES_PER_BIN)
            result.metadata.setdefault("cache_tag", pgas_run_tag)
            return result

    print(
        f"  [PGAS] running inference (target={downsample_label}, ms={maxspikes}, "
        f"resample={pgas_resample_fs:.1f}Hz)…"
    )
    result = run_pgas_inference(
        trials=trials_resampled,
        dataset_tag=pgas_run_tag,
        output_root=PGAS_OUTPUT_ROOT,
        constants_file=constants_path,
        gparam_file=PGAS_GPARAM,
        niter=PGAS_NITER,
        burnin=PGAS_BURNIN,
        recompute=True,
    )
    if refine_bins:
        result = refine_pgas_result(
            base_result=result,
            trials_resampled=trials_resampled,
            dataset_tag=pgas_run_tag,
            output_root=PGAS_OUTPUT_ROOT,
            constants_file=constants_path,
            gparam_file=PGAS_GPARAM,
            niter=PGAS_NITER,
            burnin=PGAS_BURNIN,
            base_fs=pgas_resample_fs,
        )
    result.metadata.setdefault("niter", PGAS_NITER)
    result.metadata.setdefault("burnin", PGAS_BURNIN)
    result.metadata.setdefault("config", ensure_serializable(config))
    result.metadata["maxspikes"] = maxspikes
    result.metadata["maxspikes_per_bin"] = PGAS_MAX_SPIKES_PER_BIN
    result.metadata["input_resample_fs"] = pgas_resample_fs
    result.metadata["bm_sigma"] = bm_sigma
    result.metadata["cache_tag"] = pgas_run_tag
    if substeps_per_frame is not None:
        result.metadata["substeps_per_frame"] = substeps_per_frame
    if physics_frequency_hz is not None:
        result.metadata["physics_frequency_hz"] = physics_frequency_hz
    save_method_cache("pgas", pgas_run_tag, result, config, trace_hash)
    return result


def fetch_ens2_result(
    dataset_tag: str,
    raw_time_stamps: np.ndarray,
    raw_traces: np.ndarray,
    neuron_type: str,
    downsample_label: str,
    use_cache: bool,
) -> MethodResult:
    trace_hash = hash_series(raw_time_stamps.ravel(), raw_traces.ravel())
    config = {
        "neuron_type": neuron_type,
        "pretrained_dir": str(ENS2_PRETRAINED),
        "downsample_target": downsample_label,
    }
    result: Optional[MethodResult] = None
    if use_cache:
        result = load_method_cache("ens2", dataset_tag, config, trace_hash)
        if result:
            return result

    print(f"  [ENS2] running inference (type={neuron_type})…")
    result = run_ens2_inference(
        raw_time_stamps=raw_time_stamps,
        raw_traces=raw_traces,
        dataset_tag=dataset_tag,
        pretrained_dir=ENS2_PRETRAINED,
        neuron_type=neuron_type,
    )
    result.metadata.setdefault("config", ensure_serializable(config))
    save_method_cache("ens2", dataset_tag, result, config, trace_hash)
    return result


def fetch_cascade_result(
    dataset_tag: str,
    trials: Sequence,
    downsample_label: str,
    use_cache: bool,
) -> MethodResult:
    trials = list(trials)
    trials_resampled = resample_trials_to_fs(trials, CASCADE_RESAMPLE_FS)
    time_flat, trace_flat = flatten_trials(trials_resampled)
    trace_hash = hash_series(time_flat, trace_flat)
    label_token = format_tag_token(downsample_label)
    cascade_cache_tag = f"{dataset_tag}_s{label_token}_rs{CASCADE_RESAMPLE_TOKEN}"
    config = {
        "model_name": CASCADE_MODEL,
        "downsample_target": downsample_label,
        "input_resample_fs": CASCADE_RESAMPLE_FS,
    }
    result: Optional[MethodResult] = None
    if use_cache:
        result = load_method_cache("cascade", cascade_cache_tag, config, trace_hash)
        if result:
            result.metadata.setdefault("input_resample_fs", CASCADE_RESAMPLE_FS)
            result.metadata.setdefault("cache_tag", cascade_cache_tag)
            return result

    print("  [CASCADE] running inference…")
    result = run_cascade_inference(
        trials=trials_resampled,
        dataset_tag=cascade_cache_tag,
        model_folder=CASCADE_MODEL_FOLDER,
        model_name=CASCADE_MODEL,
    )
    result.metadata.setdefault("config", ensure_serializable(config))
    result.metadata["input_resample_fs"] = CASCADE_RESAMPLE_FS
    result.metadata["cache_tag"] = cascade_cache_tag
    save_method_cache("cascade", cascade_cache_tag, result, config, trace_hash)
    return result


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
        "--pgas-refine-bins",
        action="store_true",
        help="Enable adaptive PGAS bin refinement on high-activity windows.",
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
    if args.pgas_substeps_per_frame is not None and args.pgas_substeps_per_frame < 1:
        parser.error("--pgas-substeps-per-frame must be >=1 when provided.")
    if args.pgas_physics_freq is not None and args.pgas_physics_freq <= 0:
        parser.error("--pgas-physics-freq must be positive when provided.")

    pgas_resample_fs = args.pgas_resample_fs or PGAS_RESAMPLE_FS
    pgas_resample_token = format_tag_token(f"{pgas_resample_fs:g}")
    substep_token = None
    if args.pgas_substeps_per_frame is not None:
        substep_token = f"spf{args.pgas_substeps_per_frame}"
    elif args.pgas_physics_freq is not None:
        substep_token = f"pfreq{format_tag_token(f'{args.pgas_physics_freq:g}')}"
    run_resample_tag = f"pgas{pgas_resample_token}_cascade{CASCADE_RESAMPLE_TOKEN}"
    if substep_token:
        run_resample_tag = f"{run_resample_tag}_{substep_token}"
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

            bm_sigma_series: Optional[Tuple[np.ndarray, np.ndarray]] = None
            if trials_for_methods:
                resampled_full = resample_trials_to_fs(trials_for_methods, pgas_resample_fs)
                full_time_flat, full_trace_flat = flatten_trials(resampled_full)
                mask = build_low_activity_mask(full_time_flat, spike_times, args.bm_sigma_spike_gap)
                if np.count_nonzero(mask) >= 2:
                    bm_sigma_series = (full_time_flat[mask], full_trace_flat[mask])

            trials_for_pgas = trim_trials_by_edges(trials_for_methods, dataset_edges)

            pgas_result = fetch_pgas_result(
                dataset_tag,
                trials_for_pgas,
                dataset_edges,
                downsample_label,
                target,
                raw_fs,
                use_cache=args.use_cache,
                refine_bins=args.pgas_refine_bins,
                pgas_resample_fs=pgas_resample_fs,
                maxspikes_override=pgas_maxspikes_override,
                bm_sigma_series=bm_sigma_series,
                substeps_per_frame=args.pgas_substeps_per_frame,
                physics_frequency_hz=args.pgas_physics_freq,
                c0_is_first_y=args.pgas_c0_first_y if hasattr(args, "pgas_c0_first_y") else None,
            )
            pgas_result.metadata["window_edges"] = dataset_edges.tolist()

            ens2_result: Optional[MethodResult] = None
            if args.skip_ens2:
                print("  [ENS2] skipping inference (--skip-ens2).")
            else:
                ens2_result = fetch_ens2_result(
                    dataset_tag,
                    time_stamps,
                    dff,
                    args.neuron_type,
                    downsample_label,
                    use_cache=args.use_cache,
                )

            cascade_result = fetch_cascade_result(
                dataset_tag,
                trials_for_methods,
                downsample_label,
                use_cache=args.use_cache,
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
                "pgas_refined_windows": ensure_serializable(
                    pgas_result.metadata.get("pgas_refined_windows", [])
                ),
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
                "pgas_refine_bins": bool(args.pgas_refine_bins),
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
