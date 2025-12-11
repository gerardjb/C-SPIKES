# C-SPIKES usage guide

This repository bundles multiple spike-inference backends (PGAS, ENS2, CASCADE) and a lightweight Python API for running and comparing them on your own calcium imaging data.

## Data expectations
- Input files: MATLAB `.mat` containing at least `time_stamps` (trials × samples) and `dff` (trials × samples). Optional `spike_times` (1D, seconds) enables correlation against ground truth.
- Optional per-trial windows: an `edges` array (shape n_trials × 2, seconds) to trim data before inference. See `extract_time_stamp_edges.py` for generating these from existing recordings.

## Core Python API
All reusable pieces live under `c_spikes/inference`:
- `workflow.run_inference_for_dataset(cfg, …)` orchestrates loading a dataset, optional downsampling, running PGAS/ENS2/CASCADE, computing correlations, and returning `MethodResult` objects plus summary metadata.
- `types.py`: `TrialSeries`, `MethodResult`, hashes/serialization helpers.
- `smoothing.py`: mean downsampling and resampling utilities.
- `pgas.py`: PGAS config (`PgasConfig`), runner, and PGAS-specific helpers (trim by edges, load trajectories).
- `ens2.py`, `cascade.py`: wrappers for ENS2 and CASCADE with caching.
- `eval.py`: ground-truth series building, correlation, resampling utilities.

### Minimal example
```python
from pathlib import Path
from c_spikes.inference.workflow import DatasetRunConfig, MethodSelection, SmoothingLevel, run_inference_for_dataset

cfg = DatasetRunConfig(
    dataset_path=Path("data/my_data/my_recording.mat"),
    smoothing=SmoothingLevel(target_fs=30.0),   # None -> raw
    selection=MethodSelection(run_pgas=True, run_ens2=True, run_cascade=True),
    pgas_resample_fs=None,                      # None => use native rate for PGAS
    cascade_resample_fs=30.0,                   # override CASCADE input rate if needed
    edges=None,                                 # optional per-trial windows
)
outputs = run_inference_for_dataset(
    cfg,
    pgas_constants=Path("parameter_files/constants_GCaMP8_soma.json"),
    pgas_gparam=Path("src/c_spikes/pgas/20230525_gold.dat"),
    pgas_output_root=Path("results/pgas_output/my_runs"),
    ens2_pretrained_root=Path("results/Pretrained_models/ens2_published"),
    cascade_model_root=Path("results/Pretrained_models"),
)
print(outputs["correlations"])
```

### Caching
Each backend caches results under `results/inference_cache/<method>/<dataset_tag>/<hash>.{mat,json}`. Reuse by setting `use_cache=True` in configs. PGAS trajectories are also written under `results/pgas_output/<tag>` for reconstruction.

## Demo script
Run `scripts/demo_compare_methods.py` to:
- Load a user-specified `.mat` file.
- Optionally trim to a window (edges file or start/end times).
- Run PGAS/ENS2/CASCADE with configurable smoothing/downsampling.
- Print correlations and plot overlays (spike_prob + discrete spikes).

Example:
```bash
python scripts/demo_compare_methods.py \
  --dataset data/my_data/my_recording.mat \
  --smoothing 30 \
  --pgas-resample 120 \
  --cascade-resample 30 \
  --edges-file results/excitatory_time_stamp_edges.npy
```

## Notes on defaults
- Smoothing: `SmoothingLevel(target_fs=None)` keeps native rate; a number (e.g., 30) down-samples before inference and defines the reference grid.
- PGAS resample: `None` uses native sampling; set explicitly (e.g., 120) to force resampling.
- CASCADE resample: defaults to 30 Hz to match the shipped model; override only if using a compatible model.
- ENS2 uses your provided traces as-is; choose `neuron_type` (`Exc`/`Inh`) to select the checkpoint.

## Where to look next
- `c_spikes/inference/workflow.py` for the end-to-end runner.
- `c_spikes/inference/pgas.py` for PGAS-specific knobs and trajectory loading.
- `inference_cache_compare.ipynb` for quick cache comparisons/plots.
