# C-SPIKES usage guide

The C-SPIKES (**C**alcium **S**pike **P**rocessing using **I**ntegrated **K**inetic **E**stimation and **S**imulation) repository bundles multiple spike-inference backends (PGAS, ENS2, CASCADE) and a Python API for running and comparing them on your own calcium imaging data.

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

### CPU/GPU PGAS builds
This repo now builds **CPU** and (optionally) **GPU** PGAS backends side-by-side:

- CPU module: `c_spikes.pgas.pgas_bound_cpu` (always built)
- GPU module: `c_spikes.pgas.pgas_bound_gpu` (built when Kokkos CUDA is available)
- Default import: `c_spikes.pgas.pgas_bound` (a shim that picks GPU if available, else CPU)

**Build options (scikit-build-core)**:

```bash
# AUTO (default): build GPU only if Kokkos CUDA is enabled
pip install -ve .

# Force GPU build (error if CUDA/Kokkos CUDA not enabled)
pip install -ve . --config-settings=cmake.args="-DPGAS_BUILD_GPU=ON"

# CPU only
pip install -ve . --config-settings=cmake.args="-DPGAS_BUILD_GPU=OFF"
```

**Runtime selection**:
- Default: GPU if `pgas_bound_gpu` imports, otherwise CPU.
- Override with `C_SPIKES_PGAS_BACKEND=cpu|gpu`, e.g.:
  ```bash
  C_SPIKES_PGAS_BACKEND=cpu python scripts/demo_compare_methods.py ...
  ```
- CPU code runs the OpenMP framework. Thread count can be specified as, e.g.:
  ```bash
  export OMP_NUM_THREADS=4
  ```

If you previously installed an older `pgas_bound` extension, remove it and rebuild so the shim
module can take effect.

## Pretrained models
This repo ships pretrained model bundles under `Pretrained_models/` at the repo root:
- ENS2 published checkpoints: `Pretrained_models/ens2_published/`
- CASCADE Universal (30 Hz): `Pretrained_models/Cascade_Universal_30Hz/`

Some entrypoints (notably `run_pipeline.py` and CASCADE in `scripts/demo_compare_methods.py`) still
default to looking under `results/Pretrained_models/`. If you keep models in `Pretrained_models/`
and don’t want to edit code, the simplest fix is a symlink:
```bash
mkdir -p results
ln -s ../Pretrained_models results/Pretrained_models
```

## Data expectations
- Input files: MATLAB `.mat` containing at least `time_stamps` (trials × samples, seconds) and `dff` (trials × samples). NaN padding is OK (it’s dropped per trial).
- Optional ground truth spikes: `ap_times` (1D, seconds). If you don’t have GT, store an empty array; correlations-to-GT will be unavailable/NaN.
- Optional per-trial windows: an `edges` array (shape n_trials × 2, seconds) to trim data before inference. See `extract_time_stamp_edges.py` for generating these from existing recordings.

## GUI usage
The gui is the primary entry point for the majority of the functionality of the codebase for what most people will use it for. It functions in both the cpu-only or gpu builds and allows you to perform inference on your dataset with our biophysical methods as well as other methods (only CASCADE and ENS2 as MLspike is implemented in Matlab) we investigated in the original publication. Core tasks include:
- Edge selection (a module that allows selection of subsets of data from every epoch. This is useful for reducing run times for the biophys_smc method, which can take a long time to run on longer datasets)
- Spike Inference (a module that performs spike prediction on your data with our biophysical methods as well as CASCADE and ENS2 methods)

To launch the desktop GUI:
```bash
python scripts/c_spikes_gui.py
```

### Spike Inference tab
This tab contains panels that allow selection of which methods (i.e., BiophysSMC, BiophysML, CASCADE, ENS2) you'd like to run on your data as well as panels for selecting specific Pretrained models for the supervised methods (BiophysML, CASCADE, ENS2) and hyperparameters for the BiophyhsSMC method.
- **Dataset**: select a directory containing `.mat` files.
- **Epoch**: navigate epochs within files (multi-epoch `.mat` supported).
- **Methods**: toggle `biophys_smc` (PGAS), `BiophysML`, `CASCADE`, `ENS2`.
- **Models**:
  - CASCADE models: `Pretrained_models/CASCADE/<model_name>/`
  - ENS2 models: `Pretrained_models/ENS2/<model_dir>/` containing `exc_ens2_pub.pt` or `inh_ens2_pub.pt`
  - BiophysML: `Pretrained_models/BiophysML/<model_dir>/` (auto-detects CASCADE vs ENS2 based on contents)
- **Run tag**: outputs are organized under `data_dir/spike_inference/<run_tag>/`.
- **Use cache**: reuses cached inference under `data_dir/spike_inference/<run_tag>/inference_cache/`.
- **Use edges**: apply an edges file (selected in the Dataset panel) when running PGAS.

### Biophys ML tab
- **Run tag**: outputs are organized under `data_dir/biophys_ml/<run_tag>/`.
- **Use cache**: PGAS cell-parameter inference can reuse cache under `data_dir/biophys_ml/<run_tag>/inference_cache/`.
- **Synthetic config**: use `Edit config` to edit/save run-scoped synthetic settings.
- **Load last synthetic config for this run**: restores `data_dir/biophys_ml/<run_tag>/biophys_ml/synthetic_config.json` into the editor.

### Edge Selection tab
- Select a dataset directory and epoch.
- Set an **epoch width** (seconds), then click the trace to define `[start, start+width]` (snapped to nearest time bins).
- Edges are saved to `data_dir/edges/edges.npy` after each change.
- Reopening the same dataset auto-loads the most recent `edges*.npy` file in `data_dir/edges/`.

### GUI data validation
Before loading data in the GUI, you can validate a directory of `.mat` files:
```bash
python scripts/validate_gui_mat.py --data-dir data/my_data
```
Add `--deep` to load arrays and report basic time/spike statistics (slower on large files):
```bash
python scripts/validate_gui_mat.py --data-dir data/my_data --deep
```
The validator checks required keys (`time_stamps`, `dff`) and reports optional `ap_times` coverage.

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

### Export downsampled datasets (for external tools)
If you want to run external methods (e.g., MLspike) on the same downsampled inputs used by this repo
(10Hz/30Hz smoothing), export a downsampled copy of a dataset directory:
```bash
PYTHONPATH=src python scripts/export_downsampled_mat_dir.py \
  --data-root data/janelia_8f/excitatory \
  --out-root results/gt_downsampled/janelia_8f_excitatory \
  --smoothing-level 30Hz --smoothing-level 10Hz
```
Optional: apply per-trial windows from an edges dict (`.npy`) before exporting:
```bash
PYTHONPATH=src python scripts/export_downsampled_mat_dir.py \
  --data-root data/janelia_8f/excitatory \
  --out-root results/gt_downsampled/janelia_8f_excitatory_windowed \
  --smoothing-level 30Hz \
  --edges-path results/excitatory_time_stamp_edges.npy
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

### Example: batch PGAS on jGCaMP8m
If you have jGCaMP8m-formatted datasets under `data/janelia_8m/excitatory/`, you can run PGAS across the whole directory via the batch CLI:
```bash
python -m c_spikes.cli.run \
  --data-root data/janelia_8m/excitatory \
  --edges-path results/excitatory_jG8m_edges_2000pts.npy \
  --pgas-constants parameter_files/constants_GCaMP8m_soma.json \
  --pgas-gparam src/c_spikes/pgas/20251207_jG8m_params.dat \
  --pgas-output-root results/pgas_output/j8m_base \
  --pgas-bm-sigma auto \
  --method pgas \
  --smoothing-level raw
```
Notes:
- `--smoothing-level raw` means “native sampling rate”; omit `--smoothing-level` to run `raw`, `30Hz`, and `10Hz`.
- Remove `--method pgas` to run ENS2/CASCADE too (requires the `results/Pretrained_models` symlink above if you keep models in repo-root `Pretrained_models/`).
- To run a *custom* ENS2 checkpoint, pass either `--ens2-pretrained-root Pretrained_models/<model_name>` or resolve by sweep tag via `--ens2-model-tag <tag> --ens2-model-root Pretrained_models`.
- If your CASCADE models live outside `results/Pretrained_models`, set `--cascade-model-root Pretrained_models`.

## Distill PGAS → custom ENS2 (synthetic training)
Once you have one or more `param_samples_*.dat` files, you can generate synthetic ground-truth datasets and train a custom ENS2 checkpoint:

```bash
python scripts/demo_pgas_to_ens2.py \
  --param-samples results/pgas_output/my_run/param_samples_<tag>.dat \
  --model-root Pretrained_models \
  --model-name ens2_custom_my_run \
  --train-ens2
```

Repeat `--param-samples ...` to train on multiple cell parameter sets (each will generate its own `results/Ground_truth/synth_*` directory).
Add `--run-compare --dataset <path.mat>` to automatically run a quick stock-vs-custom ENS2 comparison after training.
If you didn’t create the symlink above, also pass `--stock-ens2-root Pretrained_models/ens2_published`.

Useful parameters when matching your dataset’s spike statistics:
- `--burnin` (discard early PGAS samples, we find ~100 is typically enough to get a stable posterior, but plotting parameter values against iterations can reveal if you need more/less on your own data)
- `--spike-rate` and `--spike-params <smooth_sec> <duty_fraction>` (note: realized mean firing rate depends on the generator; verify on outputs)
- `--noise-dir`, `--noise-fraction`, `--noise-seed` / `--noise-seed-base`, `--noise-target-fs`
- `--gparam-path` (sensor-specific fluorescence model used by `syn_gen`)
- `--synth-tag-suffix` (avoid reusing `results/Ground_truth/synth_*` directories across sweeps)
- `--force-synth` (allow overwriting an existing `synth_*` output directory)
- `--no-seed-spikes` (legacy syn_gen behavior; non-reproducible spike draws even when `--noise-seed` is set)

Quick QC on a generated synthetic directory (timebase + firing-rate stats):
```bash
python scripts/inspect_synth_dir.py --synth-dir results/Ground_truth/synth_<tag>
```

Outputs:
- Synthetic datasets: `results/Ground_truth/synth_<tag>/...`
- Custom ENS2 checkpoint: `<model-root>/<model-name>/exc_ens2_pub.pt` (or `inh_...`)
- Provenance: `<model-root>/<model-name>/ens2_manifest.json`

## Evaluate a custom ENS2 (single file or whole directory)
Single dataset quick check (runs stock + custom ENS2 and prints correlations):
```bash
python scripts/demo_compare_methods.py \
  --dataset data/my_data/my_recording.mat \
  --ens2-pretrained-root Pretrained_models/ens2_published \
  --ens2-custom-root Pretrained_models/<model-name> \
  --skip-pgas --skip-cascade
```

Directory evaluation (ENS2-only, writes `summary.json` + `summary.csv`):
```bash
python scripts/eval_ens2_dir.py \
  --ens2-root Pretrained_models/<model-name> \
  --dataset-dir data/my_data \
  --out-dir results/ens2_eval/<model-name>__my_data \
  --corr-sigma-ms 50 \
  --no-cache
```

Add `--smoothing <Hz>` (e.g. `--smoothing 30`) to evaluate on downsampled inputs.

### Trial-wise correlations
Batch runs can also store per-trial correlations (one correlation per epoch/window):
```bash
python -m c_spikes.cli.run --trialwise-correlations ...
```
For existing results, you can compute these retroactively from cached outputs:
```bash
python scripts/trialwise_correlations.py --data-root data/my_data --eval-root results/full_evaluation --edges-path <edges.npy>
```

You can also recompute the `summary.json` correlations in-place (without rerunning inference) using the cached outputs
referenced by each `comparison.json`:
```bash
python -m c_spikes.cli.run \
  --data-root data/janelia_8f/excitatory \
  --edges-path results/excitatory_time_stamp_edges.npy \
  --output-root results/full_evaluation_by_run \
  --run-tag base \
  --dataset jGCaMP8f_ANM471993_cell01 \
  --smoothing-level 10Hz \
  --method pgas --method ens2 --method cascade \
  --cascade-model-name Cascade_Universal_30Hz \
  --corr-sigma-ms 50 \
  --eval-only
```

### Visualization (trialwise `viz` module)
This repo now includes a small, notebook-friendly visualization module under `src/c_spikes/viz/` that builds figures from:
- `results/trialwise_correlations.csv` (from `scripts/trialwise_correlations.py` or `--trialwise-correlations` runs), and
- cached method outputs in `results/inference_cache/<method>/...`.

Two primary entrypoints live in `src/c_spikes/viz/trialwise_plots.py`:
- `plot_corr_vs_sigma(...)`: Matlab-like “shaded error bar” (mean ± SEM) of correlation vs `corr_sigma_ms`.
- `plot_trace_panel(...)`: stacked fluorescence + GT + normalized/offset method traces for a representative trial, with per-trace `r`.

CLI wrappers:
```bash
python scripts/plot_trialwise_corr_vs_sigma.py --csv results/trialwise_correlations.csv --out results/trialwise_corr_vs_sigma.png
python scripts/plot_trialwise_trace_panel.py --csv results/trialwise_correlations.csv --data-root data/janelia_8m/excitatory --dataset <dataset_stem> --out results/trialwise_trace_panel.png
```

Notebook template:
- `notebooks/trialwise_visualizations.ipynb` shows how to import the functions (without installing the package) and tweak parameters interactively.

Note: this section will need to be updated as new methods are brought online (e.g. `PGBAR`, `MLspike`) so labels/colors and method-to-run-tag conventions stay consistent.

### Import external (MATLAB) method outputs
If you have spike-probability traces generated outside this repo (e.g. MATLAB), you can import them into the
existing cache + `full_evaluation` layout so the plotting/eval scripts can pick them up:
```bash
PYTHONPATH=src python scripts/import_matlab_cache.py \
  --pred-path /path/to/method_outputs.mat \
  --dataset <dataset_stem> \
  --smoothing raw \
  --method mlspike \
  --run-tag matlab_mlspike \
  --data-root data/janelia_8f/excitatory
```

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
    ens2_pretrained_root=Path("Pretrained_models/ens2_published"),
    cascade_model_root=Path("Pretrained_models"),
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
- CASCADE resample: defaults to the input sampling rate (no forced resampling); use `--cascade-resample` (demo) or `--cascade-resample-fs` (batch CLI) to force a specific Hz.
- ENS2 uses your provided traces as-is; choose `neuron_type` (`Exc`/`Inh`) to select the checkpoint.

## Where to look next
- `c_spikes/inference/workflow.py` for the end-to-end runner.
- `c_spikes/inference/pgas.py` for PGAS-specific knobs and trajectory loading.
- `inference_cache_compare.ipynb` for quick cache comparisons/plots.
