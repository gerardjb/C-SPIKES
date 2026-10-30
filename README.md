# C-SPIKES

C-SPIKES provides calcium-imaging spike inference with biophysical particle Gibbs
sampling (**BiophysSMC / PGAS**), synthetic-data generation and supervised-model
training (**BiophysML**), and adapters for ENS2, CASCADE and OASIS. It includes a
Python API, command-line tools and a desktop GUI.

## System requirements and installation

The portable build targets **Linux and Windows x86-64**, with separate CPU and
CUDA wheels for CPython 3.11–3.14. It uses scikit-build-core, CMake, pinned vcpkg
native dependencies, uv and cibuildwheel. macOS and other architectures are deferred.
See the [build and release guide](docs/build-and-release.md) for prerequisites,
artifact selection, the CI matrix and validation limits.

From a complete source checkout with the native compiler prerequisites installed:

```text
uv sync --locked --python 3.12 --no-install-project
uv run --no-sync python scripts/build_native.py --backend cpu --test
```

For a CUDA build, provision the pinned toolkit and select the GPU architecture
explicitly (this example targets an RTX 2080 Ti):

```text
uv run --no-sync python scripts/prepare_cuda.py
uv run --no-sync python scripts/build_native.py --backend cuda --cuda-root build/tools/cuda --cuda-arch TURING75 --test
```

Install the repaired wheel produced under `dist/`. GUI, ENS2 and CASCADE framework
requirements are optional extras: `[gui]`, `[ens2]`, `[cascade]`, or `[all]` for the
previous full application stack. Core wheel tests do not establish full GUI or
training compatibility. OASIS builds from checked-in generated C++ without Cython.
TensorFlow/CASCADE currently requires Python 3.11–3.13; core PGAS/OASIS builds do
not install TensorFlow or PyTorch and also support Python 3.14.

GPU PGAS is the intended inference/performance path. CPU PGAS is available for
small smoke checks; do not assume it is a practical substitute for long GPU runs.
The automatic `pgas_bound` shim prefers GPU when importable. Set
`C_SPIKES_PGAS_BACKEND=gpu` to require that backend for an inference command.

The [historical Linux provider](docs/numerical-provider-linux.md) and
[native-core installation](docs/native-core-installation.md) still require the
original hash-qualified dependency bundle. Selecting them does not restore the
handoff's older Kokkos version. The new provider and Kokkos upgrade need separate
scientific acceptance; passing build and regression checks does not establish
reproduction of the handoff's exact numerical references.

## Demo with supplied data

After installation, the existing comparison script can run **OASIS only** on the
first epoch of a supplied recording without loading pretrained ML models:

```bash
python scripts/demo_compare_methods.py \
  --dataset sample_data/janelia_8f/excitatory/jGCaMP8f_ANM471993_cell01.mat \
  --epoch-start 0 --epoch-stop 1 \
  --run-oasis --skip-pgas --skip-ens2 --skip-cascade --no-cache
```

The script prints `Methods run: ['oasis']` and correlations when ground-truth spikes
are present. Add `--plot` for an interactive fluorescence/ground-truth/inference
overlay. With `--no-cache`, this example does not create a method cache; omitting
it enables caching. OASIS's default output is continuous nonnegative event amplitude,
not a calibrated spike probability or count; optional binary event support uses
an explicit threshold.

The command and sample path are present in this source. A fresh end-to-end runtime
and numerical reference output for this exact demo command have not yet been
recorded; no completion-time promise is made. This demonstration is separate from
GPU PGAS inference and full BiophysML training. See [detailed workflows](docs/usage-guide.md)
for those entrypoints and required inputs.

## Use with your own data

Input requirements differ by entrypoint:

| Path | Input contract |
|---|---|
| Batch CLI and comparison demo | MATLAB files read by `scipy.io.loadmat`, with numeric `time_stamps` and `dff` arrays shaped `(trials, samples)`, and an `ap_times` key. Times are in seconds. Use an empty `ap_times` array when ground truth is unavailable. |
| Desktop GUI | `time_stamps` and `dff`, with optional `ap_times`; supports epoch extraction and a separate h5py fallback for MATLAB v7.3. This is not a general-purpose HDF5 importer. |
| External prediction import | `.mat` or `.npz` containing timestamps and predicted `spike_prob`, with configurable key names; imports predictions into the cache/evaluation layout rather than raw arbitrary imaging formats. |

Nonfinite time/fluorescence pairs are filtered by the workflow. Prepare aligned
arrays and check sampling/model requirements: OASIS requires regular within-trial
sampling, and PGAS requires appropriate indicator and cell/noise parameters.
Providing correctly named arrays does not establish scientific suitability.

For example, export a compatible batch input:

```python
import numpy as np
from scipy.io import savemat

savemat("my_recording.mat", {
    "time_stamps": time_stamps,  # trials x samples, seconds
    "dff": dff,                  # same shape
    "ap_times": np.asarray(ap_times),  # spike times in seconds, or an empty array
})
```

Inspect GUI inputs with `python scripts/validate_gui_mat.py --data-dir /path/to/data --deep`.
Launch the desktop interface with `python scripts/c_spikes_gui.py`; use its Edge
Selection tab to choose windows. The GUI launcher locates assets through the
checkout or `C_SPIKES_PROJECT_ROOT`.

| Task | Entry point / guide |
|---|---|
| Batch inference and evaluation | `python -m c_spikes.cli.run --help` or `python run_pipeline.py --help`; [detailed workflows](docs/usage-guide.md) |
| Programmatic inference | [workflow.py](src/c_spikes/inference/workflow.py), [configuration/results](src/c_spikes/inference/types.py) |
| GPU modes and reproducibility | [GPU PGAS options](docs/pgas_gpu_options.md), [validation limits](docs/pgas_gpu_validation.md) |
| Synthetic generation and model training | [BiophysML pipeline](src/c_spikes/biophys_ml/pipeline.py), `scripts/demo_pgas_to_ens2.py`, `scripts/pgas_to_ens2.py`, `scripts/pgas_to_cascade.py`; [examples](docs/usage-guide.md) |
| Downsampled MAT export | `scripts/export_downsampled_mat_dir.py --help` |
| External prediction import | `scripts/import_matlab_cache.py --help`, [import API](src/c_spikes/inference/import_external.py) |
| Plots and cached results | [visualization notebook](notebooks/trialwise_visualizations.ipynb), [cache comparison notebook](notebooks/inference_cache_compare.ipynb) |

Model assets use nested directories: `Pretrained_models/ENS2/ens2_published` and
`Pretrained_models/CASCADE/Cascade_Universal_30Hz`. Some older CLI defaults use a
different `results/Pretrained_models` layout; pass explicit model paths as described
in the workflow guide. Do not assume a single symlink repairs every entrypoint.
Synthetic generation requires native GCaMP and sensor/noise data; ENS2/CASCADE
training additionally requires their frameworks and training configuration.

A Colab demonstration and broader ingestion routines are planned. They are not
currently supplied or qualified interfaces.

## Tests, license, and citation/publication

[Test profiles](tests/README.md) separate native-core regression, Python adapters
and optional application tests. A passing native profile does not certify every
backend, model or installation environment. Optional
[observational diagnostics](docs/observational-diagnostics.md) are a build-time
choice, separate from mandatory correctness protections.

The repository includes the [GNU GPL v3 license](LICENSE). Retain applicable
third-party notices; [OASIS provenance](src/c_spikes/oasis/PROVENANCE.md) documents
its source attribution and local changes. Model/data licensing and attribution
must be considered separately from the code license.

When reporting results, identify the software version/revision, inference method,
model and parameter settings, and cite the associated publication and data/software
records as appropriate. The [publication reference](docs/PUBLICATION_REFERENCE.md)
links the existing archived materials and explains their relationship to this code.

The [Code Ocean capsule](https://codeocean.com/capsule/0036463/tree) provides the
publication's runnable workflow; associated materials are on
[Figshare](https://doi.org/10.6084/m9.figshare.28550441). This repository is not
claimed to be identical to that capsule or to reproduce every manuscript result.
See the [publication reference](docs/PUBLICATION_REFERENCE.md).
