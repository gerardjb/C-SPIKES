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

from pathlib import Path
from typing import Optional

import numpy as np

from c_spikes.utils import load_Janelia_data

# Ensure repository root is on sys.path for local imports when run as a script
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare_inference_methods import (
    TrialSeries,
    extract_trials,
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
    return float(np.max(np.abs(a - b)))


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


def main() -> None:
    data_dir = Path("data/janelia_8f/excitatory")
    dataset_files = sorted(data_dir.glob("*.mat"))
    if not dataset_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}")
    dataset_path = dataset_files[0]
    dataset_tag = dataset_path.stem

    print(f"Using dataset: {dataset_path}")
    time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))

    # ENS2: legacy vs new
    pretrained_dir = Path("results/Pretrained_models/ens2_published")
    if not pretrained_dir.exists():
        print(f"[WARN] ENS2 pretrained dir missing: {pretrained_dir} (ENS2 diff will likely fail)")

    print("\nRunning ENS2 (legacy wiring)...")
    legacy_ens2 = legacy_run_ens2(
        raw_time_stamps=time_stamps,
        raw_traces=dff,
        dataset_tag=dataset_tag,
        pretrained_dir=pretrained_dir,
        neuron_type="Exc",
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
    # Match the 30 Hz smoothing-level behavior
    trials_mean = [
        mean_downsample_trace(trial.times, trial.values, 30.0) for trial in trials_native
    ]
    trials_for_legacy = resample_trials_to_fs(trials_mean, CASCADE_RESAMPLE_FS)

    print("Running CASCADE (legacy wiring)...")
    legacy_cascade = legacy_run_cascade(
        trials=trials_for_legacy,
        dataset_tag=dataset_tag + "_test_legacy",
        model_folder=model_folder,
        model_name="Cascade_Universal_30Hz",
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
    # Pass the pre-smoothed (but not explicitly 30 Hz-resampled) trials; the new
    # wrapper will resample as needed.
    new_cascade = new_run_cascade(
        trials=trials_mean,
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

    # Legacy PGAS: explicitly resample to PGAS_RESAMPLE_FS and run the older helper.
    print("Running PGAS (legacy wiring)...")
    legacy_trials_resampled = resample_trials_to_fs(trials_for_pgas, PGAS_RESAMPLE_FS)
    legacy_pgas = legacy_run_pgas(
        trials=legacy_trials_resampled,
        dataset_tag=dataset_tag + "_pgas_legacy",
        output_root=pgas_output_root,
        constants_file=base_constants,
        gparam_file=gparam_file,
        niter=PGAS_NITER,
        burnin=PGAS_BURNIN,
        recompute=True,
        verbose=False,
    )

    # New PGAS: let the refactored wrapper handle resampling and parameter heuristics.
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
        edges=None,
        use_cache=False,
    )
    spike_times_flat = np.asarray(spike_times, dtype=np.float64).ravel()
    new_pgas = new_run_pgas(
        trials_for_pgas,
        raw_fs=raw_fs,
        spike_times=spike_times_flat,
        config=pgas_cfg,
    )

    _print_method_diff("PGAS", legacy_pgas, new_pgas)


if __name__ == "__main__":
    main()
