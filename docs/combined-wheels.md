# Combined CPU/CUDA wheels

Linux/Windows x86-64 release wheels bundle independent CPU and CUDA PGAS extensions in one
`c-spikes` distribution. Once published, install with `uv pip install c-spikes`,
or a normal dependency followed by `uv sync` in a consumer project.
See [tagged publishing](build-and-release.md#publishing-tagged-releases-to-pypi)
for release triggers and the one-time Trusted Publisher setup.

Linux ARM64 and Apple Silicon macOS wheels use the same package name and version with the CPU
OpenMP backend and OASIS only. Explicit CUDA selection there raises an error
explaining that these platforms' wheels are CPU-only.

## Implementation

Kokkos has one backend configuration per CMake build tree. With
`CSP_COMBINED_WHEEL=ON`, CMake runs two isolated builds from the same sources:

- CPU: OpenMP/Serial Kokkos, with CUDA disabled.
- GPU: CUDA-enabled Kokkos, using the existing portable Turing baseline and
  all nine native GPU targets plus PTX.

CMake installs the CPU extension, OASIS and common notices from the CPU build,
then the GPU extension and CUDA notices from the CUDA build. Scikit-build-core
packages those files into one wheel before auditwheel/delvewheel repair. No
wheel archives are merged after repair. The CPU extension has no NVIDIA driver
dependency; the GPU extension retains the host driver dependency. CUDA's runtime
remains statically linked. The CUDA child build also compiles an unused CPU
extension; it is excluded from the combined wheel.

Both extensions hide their private C++ symbols, and their pybind11 classes and
enum are registered locally to each module. They can be loaded in either order
in one Python process without sharing Kokkos initialization state or colliding
in pybind11's type registry.

## Runtime selection

```python
from c_spikes.pgas import get_backend

cpu = get_backend()           # CPU by default; no CUDA initialization.
gpu = get_backend("cuda")     # Load and initialize GPU support on demand.

# Construct models with cpu.Analyzer(...) or gpu.Analyzer(...).
# Both can coexist; selecting a module does not move existing models.
```

`get_backend("cpu")` and `get_backend("cuda")` return cached native modules.
An explicit unavailable CUDA request raises `ImportError` with guidance; CPU
execution remains usable after the failed request. The GPU extension checks for
a visible CUDA device before calling Kokkos initialization, avoiding its abort
path when no device is available.

The legacy `pgas_bound` shim now loads its selected backend only when a native
attribute is accessed. It defaults to CPU and respects
`C_SPIKES_PGAS_BACKEND=cpu|cuda`. Code that previously relied on automatic GPU
preference should select CUDA explicitly. A `from ... import Analyzer` binding
remains bound to the class chosen at that import; use `get_backend()` when
selecting backends after import. Objects belonging to a backend stay with it.

The convenience inference pipeline still uses the environment/shim selection;
the current API does not add a per-call backend argument to `PgasConfig`.

## Building and testing

After preparing the normal build dependencies and pinned CUDA toolkit:

```bash
uv run --no-sync python scripts/build_native.py \
  --backend cuda --cuda-arch PORTABLE \
  --cuda-root build/tools/cuda-12.9.1 --test
```

For the portable distribution matrix:

```bash
uv run --no-sync python scripts/build_wheels.py \
  --backend cuda --output-dir dist/combined-wheels
```

Combined wheels use the ordinary source version, without the legacy `+cu129`
suffix. Install one repaired wheel in a fresh environment for local validation.
The public package and import names remain `c-spikes` and `c_spikes`.

CI builds combined wheels by default for Linux/Windows x86-64 CPython 3.11–3.14.
All wheels run installed CPU smoke and regression tests on hosted runners. The
GPU image inspector verifies that CPU has no device code and GPU has the complete
portable target set. The combined smoke checks lazy imports and clean failure of
unavailable CUDA. Hardware validation additionally exercises CPU/GPU coexistence
in both import orders and runs the first model again after the other backend.

Source-checkout `uv sync` and `build_native.py` still default to CPU development
builds. `build_native.py --backend cuda` selects independent CPU/CUDA builds;
`build_wheels.py` defaults to that combined profile on Linux/Windows x86-64 and CPU on
Linux/macOS ARM64, with `--backend cpu` available
for CPU-only artifacts. No extra `--combined` option is needed.

The former GitHub Pages release-index implementation is preserved at `eb4c7a2`
on `build/cuda-wheel-index`. Current releases use PyPI and ordinary versions.

## Local evidence

The candidate version `0.2.0rc1` is an unpublished test version. The local Linux
build targets the development machine's glibc 2.39; portable manylinux builds
are configured in CI, and are a separate artifact qualification step.

Linux/WSL CPython 3.14.6 on the RTX 2080 Ti passed all eight native checks
(five CPU and three GPU in the two child builds), 91 native-core and 93 Python
contract tests, both NumPy/native import orders, native and forced-PTX GPU
execution, and CPU/GPU coexistence in both import orders. The same repaired
wheel passed both regression profiles and CPU inference in a plain Ubuntu
24.04 Docker container with no NVIDIA driver library. A consumer project's
`uv sync` and repeated `uv sync --locked` installed that wheel successfully.

Windows 11 CPython 3.14.6 on the same RTX 2080 Ti passed all eight native
checks, the same 91 + 93 Python tests, both NumPy/native import orders, native
and forced-PTX GPU execution, and CPU/GPU coexistence in both orders. With
GPUs hidden, CUDA selection failed cleanly and CPU inference still worked.
The CPU extension's PE imports contain no CUDA runtime or NVIDIA driver DLL.
The Windows host has an NVIDIA driver installed; the genuinely driver-free
runtime test was performed in the Linux container.

The container explicitly checked that loading `libcuda.so.1` failed. Hiding
GPUs on a driver-equipped host is an additional test, rather than the basis for
the driver-free claim. Tests use `CUDA_VISIBLE_DEVICES=-1`, because assigning
an empty environment value can remove that variable from the Windows C runtime.

Evidence is retained under `dist/combined-linux-py314/` (`validation.json`,
`driverless.json`, JUnit files and the repaired wheel), with logs under `build/`.
Windows evidence is under
`C:\Users\david\source\c-spikes-build\combined-wheel-experiment\dist\`,
with the wheel receipt and Python JUnit files in its `repaired` subdirectory.
The repaired wheel SHA-256 values are:

```text
Linux:   f91980d90db5ba643974a2dbcaa323ff4e8b1a4a749be48e08b4202d401eba8c
Windows: 2cd76baf3e0ef71e767c68275afa783eeb1a4ad16f659fb518d85712c56d6c30
```

A subsequent Linux build using the final default `--backend cuda` command
(without an opt-in combined flag) passed the same eight native and 184 Python
checks, native/forced-PTX GPU execution and CPU/GPU coexistence. Its evidence is
under `dist/combined-release-default/`, with repaired wheel SHA-256
`425d79739f8b795d5fbaa514bc70c74aef1b855a622f6862c398e83bb9f31d79`.

These checks establish build/runtime behavior for the combined package, not historical
scientific qualification or performance equivalence for full inference jobs.
