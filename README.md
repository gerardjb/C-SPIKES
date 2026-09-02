## Nature Methods software availability details (for submission)

### 1) Dependencies, operating systems, tested versions, and hardware requirements
- **Primary software dependencies**:
  - Python `>=3.8` (recommended/tested path in this repo: Python `3.10` to `3.11`)
  - NumPy, SciPy, Matplotlib, PySide6, h5py, ruamel.yaml
  - PyTorch `>=2.0`
  - TensorFlow (CPU and GPU-capable installs are supported where available)
  - Native build stack for PGAS and OASIS: `scikit-build-core`, `pybind11`, NumPy headers,
    a CMake toolchain, and a C++ compiler
  - For GPU PGAS builds: Kokkos with CUDA enabled (see `kokkos_install.md`)
- **Operating systems supported**:
  - Linux (`x86_64`) and Windows (`x86_64`) are supported for installation and CPU workflows.
  - **GPU backends are currently available on Linux only**.
- **Versions tested for this release**:
  - Linux: Ubuntu `22.04`/`24.04` (CPU; GPU path tested on Ubuntu `22.04` with CUDA-enabled Kokkos)
  - Windows: Windows `11` (CPU workflows)
- **Non-standard hardware requirements**:
  - No non-standard hardware is required for CPU workflows.
  - For GPU acceleration, an NVIDIA GPU with a compatible CUDA driver/toolchain is required (Linux only).

### 2) Typical installation time on a "normal" desktop computer
- **CPU-only install** (`conda create` + `pip install -e .`): typically **~10-25 minutes**.
- **GPU-enabled install** (including Kokkos/CUDA toolchain setup and PGAS GPU build): typically **~30-90 minutes**, depending on compiler/cache state and local CUDA setup.

# C-SPIKES usage guide

The C-SPIKES (**C**alcium **S**pike **P**rocessing using **I**ntegrated **K**inetic **E**stimation and **S**imulation) repository bundles multiple spike-inference backends (PGAS, ENS2, CASCADE, and OASIS) and a Python API for running and comparing them on your own calcium imaging data.

## Installation (build native extensions + deps)
PGAS and OASIS include compiled C++ extensions. The quickest path on Linux/HPC is:

1. Install C++ deps via vcpkg (see `kokkos_install.md` for the exact commands/pins used in this repo).
2. Install the Python package in editable mode (builds the extension):
   ```bash
   conda create -n c_spikes python=3.11
   pip install -e .
   ```

Check that both native backends import:
```bash
python -c "import c_spikes.pgas.pgas_bound as p; print('pgas_bound OK')"
python -c "import c_spikes.oasis.oasis_methods as o; print('oasis_methods OK')"
```

### OASIS build

OASIS is built by default from the checked-in generated C++ source. Normal installation neither
installs nor runs Cython. The build was audited with NumPy 1.26 and NumPy 2.x; isolated builds choose
compatible NumPy headers for the selected Python version. To intentionally build C-SPIKES without
OASIS, use:

```bash
pip install -ve . --config-settings=cmake.args="-DC_SPIKES_BUILD_OASIS=OFF"
```

A clean serialized OASIS compilation added approximately 20–21 seconds on the audited HPC build
host. An unchanged no-op target took about 0.14 seconds, so incremental build overhead is
negligible. See `src/c_spikes/oasis/PROVENANCE.md` for regeneration details and source attribution.

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
- OASIS inputs must be uniformly sampled within a trial and use a consistent effective rate across
  trials. The workflow can downsample before dispatch, but the OASIS adapter never resamples or
  treats an irregular trace as uniform implicitly.

## OASIS spike inference

OASIS is an opt-in runtime method even though its native extension is built by default. This keeps
the existing PGAS/ENS2/CASCADE comparison defaults unchanged. An OASIS-only batch run with
per-trial automatic AR(1) and noise estimation is:

```bash
python -m c_spikes.cli.run \
  --data-root data/my_data \
  --dataset my_recording \
  --smoothing-level raw \
  --method oasis \
  --oasis-ar-order 1 \
  --use-cache
```

The demo script exposes the method through `--run-oasis`:

```bash
python scripts/demo_compare_methods.py \
  --dataset data/my_data/my_recording.mat \
  --run-oasis --skip-pgas --skip-ens2 --skip-cascade
```

The lower-level framework adapter accepts already-preprocessed trials directly:

```python
from c_spikes.inference import OasisConfig, TrialSeries, run_oasis_inference

result = run_oasis_inference(
    [TrialSeries(times=trial_times, values=trial_dff)],
    OasisConfig(
        dataset_tag="my_recording",
        g=(None,),       # estimate one AR coefficient on this processed trial
        sn=None,         # estimate its noise level
        penalty=1,
        use_cache=True,
    ),
)
```

Important output semantics and limitations:

- Each trial is deconvolved independently, so calcium state never crosses gaps between epochs.
- `MethodResult.spike_prob` contains the continuous nonnegative OASIS event amplitude `s`. Despite
  the shared field name, it is not a calibrated probability or an integer spike count.
- `MethodResult.reconstruction` is the baseline-inclusive `c + b` signal.
- `discrete_spikes` is deliberately `None`; thresholding or rounding `s` into counts would require
  a separately specified event policy.
- AR coefficients in `g` are per-bin values. The default estimates one AR(1) coefficient and the
  noise level independently for every processed trial. Fixed `g` values must match the selected AR
  order.
- The framework default is convex L1 (`penalty=1`). AR(1) uses the compiled constrained solver;
  AR(2) currently preserves the comparator's Python constrained ONNLS backend.
- Cache entries live under `results/inference_cache/oasis/<dataset>_s<label>/`. Their identity
  includes exact timestamps/values, trial boundaries, sampling rates, preprocessing label, every
  solver option, and the OASIS source/adapter revision.

The implementation is based on OASIS by Johannes Friedrich and the methods described by Friedrich
and Paninski (NIPS 2016) and Friedrich, Zhou, and Paninski (PLOS Computational Biology 2017). Full
source-revision and licensing details are in `src/c_spikes/oasis/PROVENANCE.md`.

## GUI usage
The gui is the primary entry point for the majority of the functionality of the codebase for what most people will use it for. It functions in both the cpu-only or gpu builds and allows you to perform inference on your dataset with our biophysical methods as well as CASCADE, ENS2, and opt-in OASIS. MLspike remains implemented in Matlab. Core tasks include:
- Edge selection (a module that allows selection of subsets of data from every epoch. This is useful for reducing run times for the biophys_smc method, which can take a long time to run on longer datasets)
- Spike Inference (a module that performs spike prediction on your data with our biophysical methods as well as CASCADE, ENS2, and OASIS)

To launch the desktop GUI:
```bash
python scripts/c_spikes_gui.py
```

The launcher publishes its checkout through `C_SPIKES_PROJECT_ROOT`. This keeps non-editable
installs pointed at the checkout's `Pretrained_models/`, `parameter_files/`, and PGAS parameter
data instead of deriving those paths from `site-packages`. Set the variable explicitly before
launching to use a different complete checkout:

```bash
C_SPIKES_PROJECT_ROOT=/path/to/C-SPIKES python scripts/c_spikes_gui.py
```

### Spike Inference tab
This tab contains panels that allow selection of which methods (i.e., BiophysSMC, BiophysML, CASCADE, ENS2, OASIS) you'd like to run on your data as well as panels for selecting specific Pretrained models for the supervised methods (BiophysML, CASCADE, ENS2) and hyperparameters for BiophysSMC and OASIS.
- **Dataset**: select a directory containing `.mat` files.
- **Epoch**: navigate epochs within files (multi-epoch `.mat` supported).
- **Methods**: toggle `biophys_smc` (PGAS), `BiophysML`, `CASCADE`, `ENS2`, or opt-in `OASIS`.
- **Models**:
  - CASCADE models: `Pretrained_models/CASCADE/<model_name>/`
  - ENS2 models: `Pretrained_models/ENS2/<model_dir>/` containing `exc_ens2_pub.pt` or `inh_ens2_pub.pt`
  - BiophysML: `Pretrained_models/BiophysML/<model_dir>/` (auto-detects CASCADE vs ENS2 based on contents)
- **Run tag**: outputs are organized under `data_dir/spike_inference/<run_tag>/`.
- **Use cache**: reuses cached inference under `data_dir/spike_inference/<run_tag>/inference_cache/`.
- **OASIS Config**: choose AR(1)/AR(2), automatic or fixed per-bin `g`, automatic or fixed
  noise/baseline, L1/L0 penalty, coefficient optimization, and decimation. OASIS runs on the raw
  uniformly sampled GUI epoch and plots continuous event amplitude; it does not create discrete
  spike markers. A missing/stale native OASIS extension is reported for OASIS only, so other
  selected methods can still finish.
- **Use edges**: apply an edges file (selected in the Dataset panel) when running PGAS.
- **Generate sbatch**: in **PGAS Config**, `Edit Slurm Profile...` edits
  `data_dir/spike_inference/<run_tag>/slurm/slurm_profile.json`, and `Generate sbatch...` previews then writes
  `data_dir/spike_inference/<run_tag>/slurm/<job_name>.sbatch`.
  The generated command includes current dataset selection, constants file, gparam file, run tag, cache mode, and
  edges usage, and a run-scoped cache root (`data_dir/spike_inference/<run_tag>/inference_cache`). `strict_mode` in
  `slurm_profile.json` controls shell flags (`eo_pipefail` default, `euo_pipefail`,
  or `off`).

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
- Auto-calibrated `bm_sigma` is clipped with `--pgas-bm-sigma-min` and `--pgas-bm-sigma-max`.
- Auto-calibrated sigma2 can seed the inverse-gamma prior; override it with `--pgas-sigma2-target`, set the shape with `--pgas-sigma2-alpha`, or use `--pgas-sigma2-prior-strength` when alpha is implicit. The target is the prior mode, so `beta = target * (alpha + 1)`. The structure allows for different distributions for the sigma2 prior.
- Auto calibration uses robust differences by default. Use `--pgas-noise-calibration-method psd` to estimate sigma2 from Welch PSD after excluding narrowband peaks; this does not notch-filter the inference input.

Where outputs go:
- `results/inference_cache/pgas/<cache_tag>/<cache_key>.mat` stores PGAS trajectories, parameter samples, logp, post-burnin means, and MAP metadata.
- `results/pgas_output/<run>/traj_samples_<tag>.dat`, `param_samples_<tag>.dat`, and `logp_<tag>.dat` are transient by default; pass `--pgas-keep-output-dat-files` if you need the legacy raw dumps.

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
- `plot_trace_panel(...)`: stacked fluorescence + GT + normalized/offset method traces for a representative trial, with per-trace `r`. Pass `trace_data_path=...` to also export the displayed traces and metadata to `.npz`.

CLI wrappers:
```bash
python scripts/plot_trialwise_corr_vs_sigma.py --csv results/trialwise_correlations.csv --out results/trialwise_corr_vs_sigma.png
python scripts/plot_trialwise_trace_panel.py --csv results/trialwise_correlations.csv --data-root data/janelia_8m/excitatory --dataset <dataset_stem> --out results/trialwise_trace_panel.png --trace-data-out results/trialwise_trace_panel.npz
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
- `workflow.run_inference_for_dataset(cfg, …)` orchestrates loading a dataset, optional downsampling, running PGAS/ENS2/CASCADE/OASIS, computing correlations, and returning `MethodResult` objects plus summary metadata.
- `types.py`: `TrialSeries`, `MethodResult`, hashes/serialization helpers.
- `smoothing.py`: mean downsampling and resampling utilities.
- `pgas.py`: PGAS config (`PgasConfig`), runner, and PGAS-specific helpers (trim by edges, load trajectories).
- `ens2.py`, `cascade.py`, `oasis.py`: framework adapters with caching. The numerical OASIS
  implementation remains isolated under `c_spikes/oasis`.
- `eval.py`: ground-truth series building, correlation, resampling utilities.

### Minimal example
```python
from pathlib import Path
from c_spikes.inference.workflow import DatasetRunConfig, MethodSelection, SmoothingLevel, run_inference_for_dataset

cfg = DatasetRunConfig(
    dataset_path=Path("data/my_data/my_recording.mat"),
    smoothing=SmoothingLevel(target_fs=30.0),   # None -> raw
    selection=MethodSelection(
        run_pgas=True,
        run_ens2=True,
        run_cascade=True,
        run_oasis=True,
    ),
    oasis_g=(None,),                             # estimate AR(1) g per processed trial
    oasis_sn=None,                               # estimate noise per processed trial
    oasis_penalty=1,                             # framework default: L1
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
Each backend caches results under `results/inference_cache/<method>/<dataset_tag>/<hash>.{mat,json}`. Reuse by setting `use_cache=True` in configs. OASIS uses a boundary-aware trace identity because trial resets affect inference. PGAS trajectories are also written under `results/pgas_output/<tag>` for reconstruction.

## Demo script
Run `scripts/demo_compare_methods.py` to:
- Load a user-specified `.mat` file.
- Optionally trim to a window (edges file or start/end times).
- Run PGAS/ENS2/CASCADE and opt-in OASIS with configurable smoothing/downsampling.
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
The batch pipeline (`python run_pipeline.py` or `python -m c_spikes.cli.run`) can run PGAS/ENS2/CASCADE/OASIS across many `.mat` files and multiple smoothing/downsample settings:

```bash
python run_pipeline.py \
  --data-root data/my_data \
  --dataset-glob '*.mat' \
  --smoothing-level raw --smoothing-level 30Hz \
  --method pgas --method ens2 --method cascade --method oasis \
  --pgas-output-root results/pgas_output/my_run \
  --output-root results/full_evaluation/my_run
```

## Notes on defaults
- Smoothing: `SmoothingLevel(target_fs=None)` keeps native rate; a number (e.g., 30) down-samples before inference and defines the reference grid.
- PGAS resample: `None` uses native sampling; set explicitly (e.g., 120) to force resampling.
- CASCADE resample: defaults to the input sampling rate (no forced resampling); use `--cascade-resample` (demo) or `--cascade-resample-fs` (batch CLI) to force a specific Hz.
- ENS2 uses your provided traces as-is; choose `neuron_type` (`Exc`/`Inh`) to select the checkpoint.
- Correlations use exact `1 / reference_fs` ground-truth grids anchored independently to each
  evaluation epoch. Aggregate correlations concatenate the aligned epoch samples before computing
  Pearson correlation; they are not the arithmetic mean of trial-wise correlations.

## Where to look next
- `c_spikes/inference/workflow.py` for the end-to-end runner.
- `c_spikes/inference/pgas.py` for PGAS-specific knobs and trajectory loading.
- `c_spikes/inference/oasis.py` for the OASIS framework contract and cache mapping.
- `c_spikes/oasis/PROVENANCE.md` for the numerical source, local changes, and Cython regeneration.
- `inference_cache_compare.ipynb` for quick cache comparisons/plots.
