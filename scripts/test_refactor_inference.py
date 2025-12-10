#!/usr/bin/env python3
"""
Quick sanity check for the inference refactor.

This script:
  - Loads the first .mat file under data/janelia_8f/excitatory.
  - Runs ENS2 via the legacy wiring (compare_inference_methods.run_ens2_inference)
    and the new wiring (c_spikes.inference.ens2.run_ens2_inference) and compares outputs.
  - Runs CASCADE via the legacy wiring (compare_inference_methods.run_cascade_inference)
    and the new wiring (c_spikes.inference.cascade.run_cascade_inference) and compares outputs.

It prints basic shape checks and max absolute differences for the main arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from c_spikes.utils import load_Janelia_data

# Ensure repository root is on sys.path for local imports when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare_inference_methods import (
    TrialSeries,
    extract_trials,
    trim_trials_by_edges,
    run_pgas_inference as legacy_run_pgas,
    run_ens2_inference as legacy_run_ens2,
    run_cascade_inference as legacy_run_cascade,
)

from c_spikes.inference.types import MethodResult
from c_spikes.inference.ens2 import Ens2Config, run_ens2_inference as new_run_ens2
from c_spikes.inference.cascade import (
    CASCADE_RESAMPLE_FS,
    CascadeConfig,
    run_cascade_inference as new_run_cascade,
)
from c_spikes.inference.smoothing import mean_downsample_trace, resample_trials_to_fs
from c_spikes.inference.pgas import (
    PGAS_RESAMPLE_FS,
    PGAS_BURNIN,
    PGAS_NITER,
    PgasConfig,
    run_pgas_inference as new_run_pgas,
)


def _max_abs_diff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return None
    return float(np.max(np.abs(a[mask] - b[mask])))


def _print_method_diff(label: str, legacy: MethodResult, new: MethodResult) -> None:
    print(f"\n[{label}] legacy vs new")
    print(f"  legacy sampling_rate: {legacy.sampling_rate:.6g}")
    print(f"  new    sampling_rate: {new.sampling_rate:.6g}")
    print(f"  time_stamps shape: legacy={legacy.time_stamps.shape}, new={new.time_stamps.shape}")
    print(f"  spike_prob  shape: legacy={legacy.spike_prob.shape}, new={new.spike_prob.shape}")
    if legacy.discrete_spikes is not None or new.discrete_spikes is not None:
        print(
            f"  discrete shape: legacy="
            f"{None if legacy.discrete_spikes is None else legacy.discrete_spikes.shape}, "
            f"new={None if new.discrete_spikes is None else new.discrete_spikes.shape}"
        )

    ts_diff = _max_abs_diff(legacy.time_stamps, new.time_stamps)
    sp_diff = _max_abs_diff(legacy.spike_prob, new.spike_prob)
    ds_diff = _max_abs_diff(legacy.discrete_spikes, new.discrete_spikes)

    print(f"  max |time_stamps diff| : {ts_diff}")
    print(f"  max |spike_prob  diff| : {sp_diff}")
    print(f"  max |discrete     diff| : {ds_diff}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check legacy vs refactored inference paths.")
    parser.add_argument(
        "--edges-file",
        type=Path,
        default=Path("results/excitatory_time_stamp_edges.npy"),
        help="Optional npy file of edges (as produced by extract_time_stamp_edges.py). Default: results/excitatory_time_stamp_edges.npy",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Optional start time (seconds) to trim all trials (overrides edges-file).",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="Optional end time (seconds) to trim all trials (overrides edges-file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path("data/janelia_8f/excitatory")
    dataset_files = sorted(data_dir.glob("*.mat"))
    if not dataset_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")
    dataset_path = dataset_files[0]
    dataset_tag = dataset_path.stem

    print(f"Using dataset: {dataset_path}")
    time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))

    # Optional trimming via edges or explicit start/end
    edges_array: Optional[np.ndarray] = None
    if args.start_time is not None and args.end_time is not None:
        if args.end_time <= args.start_time:
            raise ValueError("end-time must be greater than start-time.")
        # Apply the same window to every trial
        edges_array = np.array([[args.start_time, args.end_time]] * time_stamps.shape[0], dtype=float)
    elif args.edges_file and args.edges_file.exists():
        edges_lookup = np.load(args.edges_file, allow_pickle=True).item()
        if dataset_tag in edges_lookup:
            edges_array = np.asarray(edges_lookup[dataset_tag], dtype=float)
        else:
            print(f"[WARN] Dataset '{dataset_tag}' not found in edges file {args.edges_file}; skipping trim.")

    # ENS2: legacy vs new (both via refactored wrapper for consistent caching)
    pretrained_dir = Path("results/Pretrained_models/ens2_published")
    if not pretrained_dir.exists():
        print(f"[WARN] ENS2 pretrained dir missing: {pretrained_dir} (ENS2 diff will likely fail)")

    print("\nRunning ENS2 (legacy tag via new wrapper)...")
    ens2_cfg_legacy = Ens2Config(
        dataset_tag=dataset_tag + "_ens2_legacy",
        pretrained_dir=pretrained_dir,
        neuron_type="Exc",
        downsample_label="raw",
        use_cache=False,
    )
    legacy_ens2 = new_run_ens2(
        raw_time_stamps=time_stamps,
        raw_traces=dff,
        config=ens2_cfg_legacy,
    )

    print("Running ENS2 (new wiring)...")
    ens2_cfg = Ens2Config(
        dataset_tag=dataset_tag,
        pretrained_dir=pretrained_dir,
        neuron_type="Exc",
        downsample_label="raw",
        use_cache=False,
    )
    new_ens2 = new_run_ens2(
        raw_time_stamps=time_stamps,
        raw_traces=dff,
        config=ens2_cfg,
    )

    _print_method_diff("ENS2", legacy_ens2, new_ens2)

    # CASCADE: legacy vs new
    model_folder = Path("results/Pretrained_models")
    if not model_folder.exists():
        print(f"[WARN] CASCADE model folder missing: {model_folder} (CASCADE diff will likely fail)")

    print("\nPreparing trials for CASCADE...")
    trials_native = extract_trials(time_stamps, dff)
    if edges_array is not None:
        try:
            trials_native = trim_trials_by_edges(trials_native, edges_array)
        except Exception as exc:
            print(f"[WARN] Failed to trim trials by edges ({exc}); proceeding without trim.")
    # Match the 30 Hz smoothing-level behavior
    trials_mean = [
        mean_downsample_trace(trial.times, trial.values, 30.0) for trial in trials_native
    ]
    if edges_array is not None:
        try:
            trials_mean = trim_trials_by_edges(trials_mean, edges_array)
        except Exception as exc:
            print(f"[WARN] Failed to trim downsampled trials by edges ({exc}); proceeding without trim.")
    trials_for_legacy = resample_trials_to_fs(trials_mean, CASCADE_RESAMPLE_FS)

    print("Running CASCADE (legacy tag via new wrapper)...")
    cascade_cfg_legacy = CascadeConfig(
        dataset_tag=dataset_tag + "_test_legacy",
        model_folder=model_folder,
        model_name="Cascade_Universal_30Hz",
        resample_fs=CASCADE_RESAMPLE_FS,
        downsample_label="30Hz",
        use_cache=False,
    )
    legacy_cascade = new_run_cascade(
        trials=trials_for_legacy,
        config=cascade_cfg_legacy,
    )

    print("Running CASCADE (new wiring)...")
    cascade_cfg = CascadeConfig(
        dataset_tag=dataset_tag + "_test_new",
        model_folder=model_folder,
        model_name="Cascade_Universal_30Hz",
        resample_fs=CASCADE_RESAMPLE_FS,
        downsample_label="30Hz",
        use_cache=False,
    )
    new_cascade = new_run_cascade(
        trials=trials_for_legacy,
        config=cascade_cfg,
    )

    _print_method_diff("CASCADE", legacy_cascade, new_cascade)

    # PGAS: legacy vs new (single trial for speed)
    base_constants = Path("parameter_files/constants_GCaMP8_soma.json")
    gparam_file = Path("src/c_spikes/pgas/20230525_gold.dat")
    pgas_output_root = Path("results/pgas_output/test_refactor")
    pgas_output_root.mkdir(parents=True, exist_ok=True)

    print("\nPreparing trials for PGAS (first trial only)...")
    if not trials_native:
        print("[WARN] No trials available for PGAS test; skipping.")
        return
    trials_for_pgas = trials_native[:1]

    # Estimate native sampling rate from the first trial
    t0 = trials_for_pgas[0].times
    dt = np.diff(t0)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        print("[WARN] Cannot estimate sampling rate for PGAS test; skipping.")
        return
    raw_fs = float(1.0 / np.median(dt))
    spike_times_flat = np.asarray(spike_times, dtype=np.float64).ravel()

    # Limit edges to the PGAS trial subset (first trial) if provided
    edges_for_pgas = None
    if edges_array is not None:
        edges_for_pgas = np.asarray(edges_array, dtype=float)
        if edges_for_pgas.shape[0] >= len(trials_for_pgas):
            edges_for_pgas = edges_for_pgas[: len(trials_for_pgas)]
        else:
            edges_for_pgas = None

    # Legacy PGAS tag using refactored wrapper to ensure caching
    print("Running PGAS (legacy tag via new wrapper)...")
    pgas_cfg_legacy = PgasConfig(
        dataset_tag=dataset_tag + "_pgas_legacy",
        output_root=pgas_output_root,
        constants_file=base_constants,
        gparam_file=gparam_file,
        resample_fs=PGAS_RESAMPLE_FS,
        niter=PGAS_NITER,
        burnin=PGAS_BURNIN,
        downsample_label="raw",
        maxspikes=None,
        bm_sigma=None,
        bm_sigma_gap_s=0.15,
        edges=edges_for_pgas,
        use_cache=False,
    )
    legacy_pgas = new_run_pgas(
        trials_for_pgas,
        raw_fs=raw_fs,
        spike_times=spike_times_flat,
        config=pgas_cfg_legacy,
    )

    # New PGAS: refactored wrapper with current tag
    print("Running PGAS (new wiring)...")
    pgas_cfg = PgasConfig(
        dataset_tag=dataset_tag + "_pgas_new",
        output_root=pgas_output_root,
        constants_file=base_constants,
        gparam_file=gparam_file,
        resample_fs=PGAS_RESAMPLE_FS,
        niter=PGAS_NITER,
        burnin=PGAS_BURNIN,
        downsample_label="raw",
        maxspikes=None,
        bm_sigma=None,
        bm_sigma_gap_s=0.15,
        edges=edges_for_pgas,
        use_cache=False,
    )
    new_pgas = new_run_pgas(
        trials_for_pgas,
        raw_fs=raw_fs,
        spike_times=spike_times_flat,
        config=pgas_cfg,
    )

    _print_method_diff("PGAS", legacy_pgas, new_pgas)


if __name__ == "__main__":
    main()
