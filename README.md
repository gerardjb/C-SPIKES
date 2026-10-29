# C-SPIKES

C-SPIKES provides calcium-imaging spike inference with biophysical particle Gibbs
sampling (**BiophysSMC / PGAS**), synthetic-data generation and supervised-model
training (**BiophysML**), and adapters for ENS2, CASCADE and OASIS. It includes a
Python API, command-line tools and a desktop GUI.

## System requirements and tested configurations

The current native PGAS/OASIS numerical-provider configuration accepts **Linux
x86-64**. It pins LP64 serial BLAS/LAPACK archives and a Fortran runtime by hash;
Windows/macOS builds are rejected by this profile. See
[numerical-provider requirements](docs/numerical-provider-linux.md).

| Component | Requirements and evidence |
|---|---|
| Python package | Metadata permits Python >=3.8. Dependencies include NumPy, SciPy, Matplotlib, PySide6, h5py, ruamel.yaml, PyTorch >=2, TensorFlow and tqdm==4.64.0. Metadata ranges are not a tested-version matrix. |
| Native build | CMake >=3.18, a compatible C++/Fortran toolchain, scikit-build-core >=0.10, pybind11 >=2.12, NumPy headers, GSL/CBLAS, JsonCpp, OpenMP and the pinned provider. Kokkos is pinned to 4.3.01. Test build hooks need CMake >=3.19. |
| Recorded CPU native installation | CPython 3.10.16, NumPy 2.1.3 and SciPy 1.15.3; PGAS/OASIS installed-prefix imports and relocation. This is narrower than full-package or wheel validation. |
| Recorded GPU PGAS configuration | NVIDIA Ampere/sm80, GNU 11.5, CUDA 12.9.1, Kokkos 4.3.01 and Release C++17. The recipe needs explicit nvcc `--expt-relaxed-constexpr`; see [CUDA build notes](docs/cuda-build-notes.md). |
| Optional GUI and training | GUI/display dependencies, compatible TensorFlow/PyTorch installations and appropriate model/data files. Native-core validation does not establish full training or GUI compatibility. |

GPU PGAS is the intended inference/performance path. CPU PGAS is available for
small smoke checks; do not assume it is a practical substitute for long GPU runs.
OASIS has a separate CPU implementation. Other GPU architectures, Python versions,
Linux distributions and redistributable wheels need their own validation. Minimum
RAM/VRAM requirements for every workflow have not been established.

## Installation

Use a complete source checkout and run commands from its root. First prepare the
compatible dependencies described in [native build prerequisites](kokkos_install.md)
and the [provider profile](docs/numerical-provider-linux.md). In particular:

- `CSP_NUMERICAL_PROVIDER_ROOT` must identify the matching hashed archives. The
  current profile also requires its pinned `/usr/lib64/libgfortran.so.3` runtime.
- The TOML build configuration includes a repository-relative vcpkg toolchain and
  CUDA/AMPERE80 choices. Review them for your environment; they do not provision
  the dependencies. Kokkos configuration may fetch source from GitHub.
- Preserve the tested CUDA compiler flags when building GPU PGAS. Production no
  longer links Armadillo; some optional compatibility tests use its headers.

After those prerequisites are prepared, an editable full-package build is:

```bash
python -m pip install -ve . \
  --config-settings=cmake.define.CSP_NUMERICAL_PROVIDER_ROOT=/path/to/verified/provider
```

Replace the provider placeholder with the prepared installation. This is a build
invocation, **not a complete fresh-machine dependency bootstrap**. For test
requirements, use `'.[test]'` instead of `.`. The full package resolves optional
application stacks too; the separate [CPU native-core install component](docs/native-core-installation.md)
installs only PGAS/OASIS and their minimal Python packages/notices.

OASIS builds from checked-in generated C++; ordinary installation does not require
Cython. `PGAS_BUILD_GPU=ON|OFF|AUTO` controls the GPU extension; `AUTO` uses the
configured Kokkos CUDA availability. Selecting `OFF` alone does not remove all CUDA
choices from the TOML toolchain. Details are in the build guides above.

Check the explicitly selected installed modules:

```bash
python -c "from c_spikes.pgas import pgas_bound_cpu; from c_spikes.oasis import oasis_methods"
# For an explicitly prepared GPU build:
python -c "from c_spikes.pgas import pgas_bound_gpu"
```

The automatic `pgas_bound` shim prefers GPU when importable. Set
`C_SPIKES_PGAS_BACKEND=gpu` to require that backend rather than silently changing
which backend an inference command uses. Clean installation time and a general
portable wheel-install recipe are not yet established for this configuration.

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