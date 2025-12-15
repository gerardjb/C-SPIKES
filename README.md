# C-SPIKES usage guide

This repository bundles multiple spike-inference backends (PGAS, ENS2, CASCADE) and a Python API for running and comparing them on your own calcium imaging data.

## Installation (build PGAS + deps)
PGAS is a compiled C++/pybind extension. The quickest path on Linux/HPC is:

1. Install C++ deps via vcpkg (see `kokkos_install.md` for the exact commands/pins used in this repo).
2. Install the Python package in editable mode (builds the extension):
   ```bash
   pip install -e .
   ```

Check that the extension imports:
```bash
python -c "import c_spikes.pgas.pgas_bound as p; print('pgas_bound OK')"
```

## Data expectations
- Input files: MATLAB `.mat` containing at least `time_stamps` (trials × samples, seconds) and `dff` (trials × samples). NaN padding is OK (it’s dropped per trial).
- Optional ground truth spikes: `ap_times` (1D, seconds). If you don’t have GT, store an empty array; correlations-to-GT will be unavailable/NaN.
- Optional per-trial windows: an `edges` array (shape n_trials × 2, seconds) to trim data before inference. See `extract_time_stamp_edges.py` for generating these from existing recordings.

### Bring your own `.mat`
Most scripts use `c_spikes.utils.load_Janelia_data`, which expects keys `time_stamps`, `dff`, and `ap_times`. If your data uses different names, the easiest path is to export a normalized `.mat` with these keys.

Example exporter (Python):
```python
import scipy.io as sio

sio.savemat("data/my_data/my_recording.mat", {
  "time_stamps": time_stamps,  # (n_trials, n_samples), seconds
  "dff": dff,                  # (n_trials, n_samples)
  "ap_times": ap_times,         # (n_spikes,), seconds (or empty)
})
```

## PGAS on your data (produce `param_samples_*.dat`)
To run PGAS and write its output files (including `param_samples_*.dat` used for distillation), the easiest entrypoint is `scripts/demo_compare_methods.py` with ENS2/CASCADE disabled:

```bash
python scripts/demo_compare_methods.py \
  --dataset data/my_data/my_recording.mat \
  --skip-ens2 --skip-cascade \
  --pgas-output-root results/pgas_output/my_run \
  --pgas-bm-sigma auto \
  --pgas-resample 120
```

Notes:
- Windowing: restrict PGAS (and correlations) to a time window using either `--start-time/--end-time` or an `--edges-file`.
- Sensor parameters: for new sensors (e.g. jGCaMP8m), point PGAS at your sensor-specific files via `--pgas-constants` and `--pgas-gparam`.

Where outputs go:
- `results/pgas_output/<run>/traj_samples_<tag>.dat` + `logp_<tag>.dat` (PGAS trajectories)
- `results/pgas_output/<run>/param_samples_<tag>.dat` (parameter samples; pass these into `demo_pgas_to_ens2.py`)

`<tag>` is the per-trial PGAS tag and typically ends in `_trial0`, `_trial1`, … (and also includes smoothing/resample/bm_sigma tokens).

## Distill PGAS → custom ENS2 (synthetic training)
Once you have one or more `param_samples_*.dat` files, you can generate synthetic ground-truth datasets and train a custom ENS2 checkpoint:

```bash
python scripts/demo_pgas_to_ens2.py \
  --param-samples results/pgas_output/my_run/param_samples_<tag>.dat \
  --model-name ens2_custom_my_run \
  --train-ens2
```

Repeat `--param-samples ...` to train on multiple cell parameter sets (each will generate its own `results/Ground_truth/synth_*` directory).
Add `--run-compare --dataset <path.mat>` to automatically run a quick stock-vs-custom ENS2 comparison after training.

Useful parameters when matching your dataset’s spike statistics:
- `--burnin` (discard early PGAS samples, we find ~100 is typically enough to get a stable posterior, but plotting parameter values against iterations can reveal if you need more/less on your own data)
- `--spike-rate` and `--spike-params <smooth_sec> <duty_fraction>`
- `--noise-dir`, `--noise-fraction`, `--noise-seed` / `--noise-seed-base`
- `--gparam-path` (sensor-specific fluorescence model used by `syn_gen`)
- `--synth-tag-suffix` (avoid reusing `results/Ground_truth/synth_*` directories across sweeps)

Outputs:
- Synthetic datasets: `results/Ground_truth/synth_<tag>/...`
- Custom ENS2 checkpoint: `results/Pretrained_models/<model-name>/exc_ens2_pub.pt` (or `inh_...`)
- Provenance: `results/Pretrained_models/<model-name>/ens2_manifest.json`

## Evaluate a custom ENS2 (single file or whole directory)
Single dataset quick check (runs stock + custom ENS2 and prints correlations):
```bash
python scripts/demo_compare_methods.py \
  --dataset data/my_data/my_recording.mat \
  --ens2-pretrained-root results/Pretrained_models/ens2_published \
  --ens2-custom-root results/Pretrained_models/<model-name> \
  --skip-pgas --skip-cascade
```

Directory evaluation (ENS2-only, writes `summary.json` + `summary.csv`):
```bash
python scripts/eval_ens2_dir.py \
  --ens2-root results/Pretrained_models/<model-name> \
  --dataset-dir data/my_data \
  --out-dir results/ens2_eval/<model-name>__my_data \
  --corr-sigma-ms 50 \
  --no-cache
```

Add `--smoothing <Hz>` (e.g. `--smoothing 30`) to evaluate on downsampled inputs.

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

## Batch runs across a directory
The batch pipeline (`python run_pipeline.py` or `python -m c_spikes.cli.run`) can run PGAS/ENS2/CASCADE across many `.mat` files and multiple smoothing/downsample settings:

```bash
python run_pipeline.py \
  --data-root data/my_data \
  --dataset-glob '*.mat' \
  --smoothing-level raw --smoothing-level 30Hz \
  --method pgas --method ens2 --method cascade \
  --pgas-output-root results/pgas_output/my_run \
  --output-root results/full_evaluation/my_run
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
