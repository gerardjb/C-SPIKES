# Linux and Windows builds

The wheel matrix targets x86-64 Linux and Windows, CPython 3.11–3.14, and
separate CPU and CUDA artifacts. macOS, ARM, musllinux, free-threaded Python and
additional Python versions are deferred. A CUDA build contains both PGAS backends
and CPU OASIS; the CPU build needs no CUDA toolkit or NVIDIA driver.

## Build inputs

- scikit-build-core builds the existing CMake project. CMake defaults to CPU.
- uv manages Python environments using the committed `uv.lock`.
- `vcpkg.json` pins GSL 2.8, JsonCpp 1.9.6, serial LP64 OpenBLAS 0.3.29 and
  reference LAPACK 3.11.0. The custom triplets use a generic CORE2 OpenBLAS target,
  rather than selecting the build machine's CPU instruction set.
- The bootstrap pins vcpkg tooling separately from its package baseline. Windows
  explicitly enables vcpkg's MinGW Fortran provider and uses MSVC for C++. A small
  overlay port builds LLVM OpenMP 19.1.7 from hash-pinned upstream archives. This
  supplies Kokkos's required OpenMP API with `/openmp:llvm`, without bundling the
  nonredistributable experimental runtime from Visual Studio.
  MSVC's `_OPENMP` macro still reports 2.0, so CMake verifies the required LLVM
  runtime API and OpenMP linkage before reporting 3.0 to Kokkos's dependency check.
- Kokkos [5.2.2](https://github.com/kokkos/kokkos/releases/tag/5.2.2) is pinned to
  commit `8e15454876298832f013ebcaf08d53825b5bda40` and uses C++20 for host and
  CUDA compilation. Its upstream Windows CUDA fixes replace the previous local
  Kokkos 4.3.01 backport; no Kokkos source patches are applied.
- CUDA compiler/runtime/CCCL component archives are pinned by SHA-256 in
  `cmake/cuda-12.6.3.json` (nvcc 12.6.85).
- Python package versions come from Git tags through setuptools-scm. Use a complete
  checkout or a generated sdist; an arbitrary source copy needs an explicit
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_C_SPIKES` value.

The sdist retains native sources, build scripts and test fixtures; large example
recordings and notebooks remain available in the Git checkout.

The portable profile uses newly built native libraries, not the exact historical
archives/runtime in the handoff. It retains the original
`linux-x86_64-lp64-serial` profile for that environment. Build, import and regression
checks do not establish numerical equivalence to its 38 exact reference runs.
The Kokkos upgrade also changes the handoff's runtime dependency. Selecting the
historical numerical-provider profile alone does not recreate that environment.
Scientific acceptance of these dependency changes remains a separate release gate.

## Local builds

Install Git and uv. On Linux install GCC/G++, gfortran, pkg-config, curl, zip and
unzip (Ubuntu: `sudo apt-get install build-essential gfortran pkg-config curl zip unzip`).
Kokkos 5 requires C++20; use GCC 10.4 or newer (the manylinux build uses GCC 12).
On Windows install Visual Studio 2022 C++ build tools and an x64 Windows SDK. Use
a native Windows checkout and PowerShell for Windows builds, and a Linux checkout
for WSL builds. Do not reuse build directories between operating systems or
compiler toolchains.

From the checkout root, on either OS:

```text
uv sync
```

The checkout pins Python 3.14 for development. `uv sync` creates `.venv`, invokes
scikit-build-core/CMake, prepares the pinned native dependencies, and installs
the CPU package in editable mode. The first run needs network access and can
take several minutes to compile native dependencies. Later runs reuse the
dependency cache under `build/` and the editable build under
`build/editable/`. On Windows, the editable install includes its native runtime
DLLs beside the extension; no developer DLL search path is required.

Python source edits take effect immediately. After native C++ or CMake edits:

```text
uv sync --reinstall-package c-spikes
```

Use `uv run python ...` or `uv run pytest ...` to run in this environment.
Optional frameworks remain opt-in, for example `uv sync --extra gui`. For
TensorFlow/CASCADE, use `uv python pin 3.13` to select Python 3.13 for the checkout,
then `uv sync --extra cascade`. This also keeps subsequent `uv run` commands on
Python 3.13. `--python 3.13` is available as a one-command override instead.

An explicit `CMAKE_TOOLCHAIN_FILE` bypasses automatic dependency setup, as does
`CSP_AUTO_DEPENDENCIES=OFF`; these are for prepared build environments. The
historical numerical-provider profile also keeps its existing explicit setup.

To build and validate a separate wheel artifact for distribution:

```text
uv run --no-sync python scripts/build_native.py --backend cpu --test
```

The script bootstraps its own vcpkg under `build/tools/`, installs dependencies
under `build/vcpkg-installed/`, builds a wheel, runs CTest, repairs the wheel and
installs it in a temporary environment outside the source tree. `--test` checks
both NumPy/native import orders, a tiny real Analyzer run, OASIS, and the
`native_core`/`python_contracts` pytest profiles. It clears dependency lookup
overrides so a developer's library path cannot conceal an incomplete wheel.
JUnit files, build settings and a successful validation receipt with wheel SHA-256
are written under `dist/`. Use `--help` for custom paths and parallelism.
The wheel scripts use Ninja on both operating systems. Plain `uv sync` uses
scikit-build-core's default generator (Visual Studio on Windows).

For CUDA, install a compatible NVIDIA display driver on the host, then:

```text
uv run --no-sync python scripts/prepare_cuda.py
uv run --no-sync python scripts/build_native.py --backend cuda --cuda-root build/tools/cuda --cuda-arch TURING75 --test
```

`TURING75` matches the local RTX 2080 Ti. The script also accepts `AMPERE80`,
`AMPERE86`, `ADA89` and `HOPPER90` for separate builds; those hardware combinations
need runtime validation. A build selects one Kokkos architecture. The first CI
CUDA artifact targets sm75; do not describe it as universally qualified for all
NVIDIA GPUs. CUDA 12.6 requires a supported host compiler (GCC <=13 or compatible
VS 2022). `prepare_cuda.py` unpacks compiler/runtime components locally and does
not install a display driver. WSL uses the Windows host's driver.

CUDA `--test` requires a working GPU and fails if the GPU extension or its kernels
cannot run. The CPU extension inside a CUDA wheel can still have CUDA runtime
dependencies; use the separate CPU artifact on machines without NVIDIA support.

Install the single repaired wheel from the chosen build's `repaired/` directory
using `uv pip install /path/to/the-wheel.whl`. CPU and CUDA variants currently have
the same distribution name/version, so install only one variant in an environment.
GUI, ENS2 and CASCADE frameworks are optional extras: `c-spikes[gui]`,
`c-spikes[ens2]`, `c-spikes[cascade]`, or `c-spikes[all]` for the previous full stack.
The native-wheel checks do not validate those optional applications.

Core installation and CI do not request any GUI or ML extras. As of 2026-10-30,
[TensorFlow](https://pypi.org/project/tensorflow/) provides Python 3.11–3.13 wheels
for the supported platforms, so use one of those versions for `[cascade]` or
`[all]`. Requesting an unavailable extra should fail dependency resolution; its
dependencies are not silently omitted on newer Python versions. Core Python 3.14
builds remain independent of TensorFlow. Python 3.15 is deferred because the core
[h5py dependency](https://pypi.org/project/h5py/) does not yet provide matching
wheels. Free-threaded interpreters are not included in this matrix.

## Local validation

The initial 2026-10-30 local run used Kokkos 5.2.2 and Python 3.12 on Ubuntu 24.04 under
WSL and native Windows, with an RTX 2080 Ti (sm75) and the pinned CUDA 12.6
components:

| Platform | Artifact | Native CTest checks | Installed-wheel Python tests |
|---|---|---:|---:|
| Linux / WSL | CPU | 5 passed | 180 passed |
| Linux / WSL | CUDA | 8 passed | 180 passed |
| Windows | CPU | 5 passed | 180 passed |
| Windows | CUDA | 8 passed | 180 passed |

The Python total is 87 `native_core` and 93 `python_contracts` tests, with no
skips. Every wheel also passed a tiny Analyzer run, OASIS and both NumPy/native
import orders. CUDA wheels exercised both CPU and GPU backends in separate
processes. Both platforms' Python 3.12 CUDA cibuildwheel artifacts were repaired
and tested on the GPU; the final manylinux CUDA build used an extracted sdist.
These are build/runtime checks, not the handoff's historical numerical
qualification or a performance benchmark. Other GPU architectures remain untested.

The Windows work required a source-built LLVM OpenMP runtime, portable C++
scratch buffers and explicit floating-point CUDA math
overloads. The latter avoids host-only MSVC integer overloads that compiled with
warnings but stalled a GPU inference smoke run. That smoke run is part of the
installed-wheel validator and has a timeout.

## CI and release artifacts

`.github/workflows/build.yml` runs on pull requests, main/master pushes, version
tags and manual dispatch. It builds and tests native CPU wheels on Ubuntu and
Windows with Python 3.14, creates an sdist, and runs cibuildwheel for Python
3.11, 3.12, 3.13 and 3.14 for both CPU and CUDA variants.
The cibuildwheel matrix repairs wheels with auditwheel/delvewheel and runs the
installed-wheel checks on CPU artifacts. CUDA jobs compile and repair; standard
hosted runners have no GPU, so CUDA runtime tests are explicitly skipped there.

The manylinux job uses the `manylinux_2_28` image selected by the pinned cibuildwheel
version and GCC toolset 12, including Fortran. Linux wheels built directly on
Ubuntu/WSL have that system's glibc baseline; they are local validation artifacts,
not replacements for the manylinux job. Auditwheel must succeed for portable
Linux distribution. NVIDIA's `libcuda.so.1` and `nvcuda.dll` drivers are explicitly
excluded from bundling, including WSL's host-driver proxy. Native dependency and CUDA runtime
notices are included in the wheels.

To reproduce cibuildwheel locally (Linux requires Docker or Podman):

```text
uv sync --locked --no-install-project --group ci
uv run --no-sync python scripts/build_wheels.py --backend cpu --output-dir dist/wheels-cpu
uv run --no-sync python scripts/build_wheels.py --backend cuda --cuda-arch TURING75 --output-dir dist/wheels-cuda
```

`--platform linux` also permits running the Linux container builds from Windows
with Docker Desktop. `--print-build-identifiers` previews the selected matrix.
To qualify one existing CUDA wheel on a GPU machine, place it alone in a directory
and run `python scripts/test_wheel.py --wheel-dir <directory> --backend cuda --skip-repair`
from the prepared uv environment. This tests the exact repaired artifact.

`.github/workflows/cuda-hardware.yml` is manual-only and requires an explicitly
registered self-hosted runner labeled `c-spikes-cuda`, plus the matching OS/x64
labels. It runs the same CTest and installed-wheel CUDA execution checks as the
local script. No runner registration is performed by this change.

The build workflows upload artifacts; they do not publish them or create GitHub
releases. `.github/workflows/release.yml` provides a separate, manual-only PyPI
publication path, described below. CPU/CUDA wheel names overlap: publish exactly
one backend for a given version. Choosing CUDA does not turn hosted build checks
into GPU runtime qualification, and pip cannot select a backend based on hardware.

The structure follows [Scientific Python cookie](https://github.com/scientific-python/cookie):
declarative packaging, an explicit supported matrix, isolated wheel tests, uv for
development/CI, and cibuildwheel for compiled extensions. See the upstream
[cibuildwheel options](https://cibuildwheel.pypa.io/en/stable/options/) and
[manylinux requirements](https://github.com/pypa/manylinux) for the platform boundary.

## PyPI Trusted Publishing

The distribution name is **c-spikes**, while Python imports use `c_spikes`.
The PyPI pending/normal Trusted Publisher must identify repository owner
`gerardjb`, repository `C-SPIKES`, workflow filename `release.yml`, and GitHub
environment `pypi`. No API token, password or recovery code belongs in GitHub.

**Merging this workflow does not publish.** Neither creating a tag nor publishing
a GitHub Release runs the publishing job. An owner must separately dispatch
`Validate or publish PyPI release` from **main**, explicitly enable its `publish`
input, and pass the environment approval. The default is validation only.

Before enabling uploads, a repository administrator must separately configure:

1. An environment named `pypi`, with a required owner/maintainer reviewer. Decide
   explicitly who may approve: if the owner also dispatches, preventing self-review
   requires a second reviewer. Prefer disabling administrator bypass.
2. A selected deployment **branch** rule allowing only `main` (the release
   workflow runs from main, with the source tag supplied as an input).
3. After reviewing those protections and the matching PyPI publisher, a repository
   Actions variable `PYPI_PUBLISHING_ENABLED` with the exact value `true`.
   Leave it absent/false until approved. It is not a secret or credential.

The enablement variable prevents the publishing job from starting before setup;
GitHub can otherwise create an unprotected environment merely by referencing a
missing name. These administrative steps are not performed by the workflow or
by opening its PR.

Release procedure:

1. Choose a version and **one** backend, CPU or CUDA. CPU is the default input;
   the release owner must confirm the choice. The other backend remains available
   as a CI artifact. Do not try to upload both variants under the same version.
2. Create an approved tag such as `v0.1.2` on a commit in main history. Existing
   `build.yml` runs on `v*` tag pushes. Wait for that entire run to succeed. This
   executes the existing wheel/native/sdist matrix and is separate build work;
   untagged development wheels cannot simply be renamed into release wheels.
3. Dispatch `release.yml` on main with the tag, the successful tag-build run ID,
   chosen backend, and `publish=false`. It retrieves only that run's
   `wheels-Linux-<backend>`, `wheels-Windows-<backend>` and `sdist` artifacts.
   It rejects PR/main-branch runs, unrelated workflows, moved tags, failed runs,
   expired/duplicate artifacts, mismatched names/versions/backends, and incomplete
   matrices. Required payload: eight repaired x86-64 CPython 3.11–3.14 wheels
   (Linux/Windows) and one source distribution. This checks metadata and contents,
   not runtime scientific behavior.
4. Review the run summary and `validated-pypi-distributions` artifact. Its
   `source.json` records the source commit/tag, source run/attempt, artifact IDs
   and GitHub digests where available, selected backend, and each file's SHA256.
5. With separate publication approval, dispatch the same inputs with
   `publish=true`. Review the new prepared manifest, then approve the `pypi`
   environment deployment. Only this final job has `id-token: write`; it has no
   checkout or build commands, rechecks the prepared bytes, and uses PyPA's
   publisher with attestations enabled. It does not create tags/releases.

Uploads can partially succeed if a network/service failure occurs. Stop and
inspect PyPI before retrying; the workflow deliberately does not use
`skip-existing` or conceal differing files. PyPI files cannot be overwritten.
The sdist is the same source for either selected wheel backend; its native build
defaults remain those documented above. No separate CUDA project or local-version
scheme is introduced here.

Offline admission tests (no native build, network or publishing):

```text
python3 -m unittest discover -s tests/packaging -p test_release_validation.py -v
```

PR checks run these controls. Opening this PR also invokes the repository's
existing `build.yml` pull-request matrix; no extra manual build dispatch is
needed. Successful offline checks are not proof that GitHub OIDC/environment
configuration or the eventual PyPI upload has been exercised.
