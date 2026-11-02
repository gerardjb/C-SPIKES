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
- CUDA compiler/runtime/CCCL and cuobjdump component archives are pinned by
  SHA-256 in `cmake/cuda-12.9.1.json` (nvcc 12.9.86). Tools are unpacked under
  `build/tools/cuda-12.9.1`; different toolkit releases use separate directories.
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
uv run --no-sync python scripts/build_native.py --backend cuda --cuda-root build/tools/cuda-12.9.1 --cuda-arch PORTABLE --test
```

`PORTABLE` is the release profile, described below. For a separately tuned local
build, the script also accepts `TURING75`, `AMPERE80`, `AMPERE86`, `ADA89`,
`HOPPER90`, `BLACKWELL100`, `BLACKWELL103`, `BLACKWELL120` and `BLACKWELL121`.
Use a fresh build directory when changing the selected Kokkos architecture or
toolkit. Individual architecture builds are for local use, not publication to
the shared release index. `prepare_cuda.py` does not install a display driver.
WSL uses the Windows host's driver.

CUDA `--test` requires a working GPU and fails if the GPU extension or its kernels
cannot run. The CPU extension inside a CUDA wheel can still have CUDA runtime
dependencies; use the separate CPU artifact on machines without NVIDIA support.

Install the single repaired wheel from the chosen build's `repaired/` directory
using `uv pip install /path/to/the-wheel.whl`. CPU and CUDA variants share the
`c-spikes` distribution and `c_spikes` import package. CUDA wheels add `+cu129`
to their version (for example `0.2.0+cu129`); install one variant per environment.
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
qualification or a performance benchmark. At that point, other GPU architectures
had not been tested.

The Windows work required a source-built LLVM OpenMP runtime, portable C++
scratch buffers and explicit floating-point CUDA math
overloads. The latter avoids host-only MSVC integer overloads that compiled with
warnings but stalled a GPU inference smoke run. That smoke run is part of the
installed-wheel validator and has a timeout.

The 2026-11-01 CUDA 12.9.1 portable-profile validation used Python 3.14.6 and
the RTX 2080 Ti with Windows driver 610.60, including WSL:

| Artifact | Native CTest checks | Installed-wheel Python tests | GPU paths |
|---|---:|---:|---|
| Linux/WSL local wheel | 8 passed | 91 native-core + 93 contracts | Native and forced PTX |
| Windows wheel | 8 passed | 91 native-core + 93 contracts | Native and forced PTX |
| Linux manylinux wheel, built from sdist | Not built by cibuildwheel | 91 native-core + 93 contracts | Native and forced PTX on WSL |

All three artifacts passed both NumPy/native import orders, CPU/GPU Analyzer
smokes, OASIS and inspection of all nine native code targets plus PTX in both
PGAS modules. The manylinux wheel has the `manylinux_2_27_x86_64` and
`manylinux_2_28_x86_64` tags. Other Python ABIs remain in the CI matrix; this
local portable-profile run exercised CPython 3.14 on Turing. The local
`0.2.0+cu129` artifacts used an explicit test version;
no `v0.2.0` tag or public release was created by these checks.

The Python regression profiles exercise the CPU backend inside the CUDA wheel;
GPU execution is covered by the native checks and Analyzer smoke runs. The
release/index suite has 22 passing tests, and workflow lint passes. A locally
served index using the real CPython 3.14 wheels was tested with unversioned
`uv pip install`, a consumer project's `uv sync`, and repeat locked sync. The
lockfile retained the advertised wheel hashes, and locked sync rejected a
deliberately modified download.

A subsequent [H100 NVL comparison](benchmarks/h100-cuda-architectures.md) on
`della-rse` tested both `PORTABLE` and `HOPPER90` builds using CUDA 12.9.1,
Kokkos 5.2.2, GCC 11.5 and Python 3.14.7. Each passed 8 native checks, 184 Python
tests, and native/forced-PTX GPU smokes. Across three short PGAS workloads,
median inference times differed by less than 1%, with no consistent advantage
for `HOPPER90`. This supports retaining the portable profile for the current
workload; it does not qualify performance on other GPU generations.

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
uv run --no-sync python scripts/build_wheels.py --backend cuda --output-dir dist/wheels-cuda
```

CUDA cibuildwheel builds default to `--cuda-arch PORTABLE`. Every repaired CUDA
wheel is inspected with cuobjdump on the build runner, checking all native images
and PTX in both PGAS modules without needing a GPU.

`--platform linux` also permits running the Linux container builds from Windows
with Docker Desktop. `--print-build-identifiers` previews the selected matrix.
To qualify one existing CUDA wheel on a GPU machine, place it alone in a directory
and run `python scripts/test_wheel.py --wheel-dir <directory> --backend cuda --skip-repair --cuda-root build/tools/cuda-12.9.1`
from the prepared uv environment. This tests the exact repaired artifact.

`.github/workflows/cuda-hardware.yml` is manual-only and requires an explicitly
registered self-hosted runner labeled `c-spikes-cuda`, plus the matching OS/x64
labels. It runs the same CTest and installed-wheel CUDA execution checks as the
local script, including forced-PTX smoke checks. `PORTABLE` is its default profile.
No runner registration is performed by this change.

`wheels.yml` is shared by CI and the CUDA release workflow so their build and
repair steps stay aligned. CPU wheels and sdists remain CI artifacts; PyPI
trusted publishing is a separate setup and is not enabled here.

## CUDA GPU coverage

The portable wheel uses CUDA 12.9.1 and the Kokkos Turing feature baseline.
Kokkos 5.2.2 permits one `Kokkos_ARCH_*` setting. All device code is therefore
compiled as `compute_75`, with native machine code generated for each target:

| GPU generation | Native code images | Runtime validation |
|---|---|---|
| Turing, including RTX 2080 Ti | sm75 | Local Linux/WSL and Windows |
| Ampere | sm80, sm86 | Compiled; hardware testing needed |
| Ada | sm89 | Compiled; hardware testing needed |
| Hopper | sm90 | Linux H100 NVL: native, forced PTX, and architecture comparison |
| Blackwell | sm100, sm103, sm120, sm121 | Compiled; hardware testing needed |

The wheel also retains `compute_75` PTX for the driver's JIT compiler. We avoid
architecture-specific `a`/`f` targets, which have narrower compatibility rules.
Both PGAS modules and Kokkos itself carry the same code images. This uses standard
nvcc code generation and does not patch Kokkos or attempt to enable several
Kokkos architecture configurations together. See NVIDIA's
[compiler documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html)
and [Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.2/blackwell-compatibility-guide/index.html).

The common feature baseline preserves the 2080 Ti path. It does not enable every
newer GPU optimization; Kokkos can warn that its compiled feature baseline differs
from the detected device. Newer GPUs still need hardware qualification and
performance measurements. Code images do not extend the Python wheel's host
platform support: ARM systems still need a separate build.

Installed-wheel CUDA tests run real Analyzer kernels normally and again with
`CUDA_FORCE_PTX_JIT=1` and the JIT cache disabled. The latter verifies the shipped
PTX path on available hardware; it does not substitute for testing a Blackwell
device. The prebuilt core wheel statically links cudart, so users need an NVIDIA
driver but no CUDA toolkit or separate NVIDIA runtime pip packages. Optional
PyTorch/TensorFlow integrations have their own runtime requirements.

Use a driver supporting CUDA 12.9 Update 1 or newer. NVIDIA lists Linux
575.57.08 and Windows 576.57 for that toolkit release; these are conservative
installation guidance, not locally tested minimums. Older CUDA 12-compatible
drivers may run native images but cannot be assumed to JIT this toolkit's PTX.
See [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html).

## Installing published wheels

The following CUDA URL becomes available after the first successful release and
Pages deployment. It is not provisioned by a local build or a pull-request run.

```text
uv pip install c-spikes --index https://gerardjb.github.io/C-SPIKES/whl/cu129/
```

This selects the latest compatible stable CUDA wheel. Add `--upgrade` when
updating an already installed CPU or CUDA variant. PyPI remains the fallback for
dependencies such as NumPy; uv's default first-index policy keeps `c-spikes` on
the CUDA index. To pin a particular release, use `c-spikes==0.2.0+cu129` instead.
With pip, pin the CUDA version when combining indexes:

```text
pip install "c-spikes==0.2.0+cu129" --extra-index-url https://gerardjb.github.io/C-SPIKES/whl/cu129/
```

`0.2.0` is an example; use a version present in the index. CPU installations use
the normal `uv pip install c-spikes` command once CPU releases exist on PyPI.

For a user's own uv project, add the dependency and named index:

```text
uv add c-spikes --index cspikes-cu129=https://gerardjb.github.io/C-SPIKES/whl/cu129/
```

Keep the generated source mapping and mark the index explicit in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "cspikes-cu129"
url = "https://gerardjb.github.io/C-SPIKES/whl/cu129/"
explicit = true

[tool.uv.sources]
c-spikes = { index = "cspikes-cu129" }
```

Commit `pyproject.toml` and `uv.lock`. Subsequent `uv sync` installs the locked
CUDA wheel; `uv sync --upgrade-package c-spikes` selects a newer release.
The index supplies SHA-256 hashes that uv records in `uv.lock` and verifies
when downloading during sync.
`explicit = true` leaves unrelated dependencies on PyPI. This is consumer-project
configuration; `uv sync` in the C-SPIKES source checkout still builds editable CPU.
See [uv's index configuration](https://docs.astral.sh/uv/concepts/indexes/).

## Publishing CUDA releases

One-time repository setup:

1. Merge the build and publishing workflows into the default branch before
   creating a release tag that includes them.
2. In **Settings → Pages → Build and deployment**, select **GitHub Actions**.
   The site is `https://gerardjb.github.io/C-SPIKES/`. These workflows deploy the
   repository's entire Pages site, so combine outputs first if it later hosts docs.
3. Ensure the `github-pages` environment permits deployment from release tags
   as well as the default branch. Existing environment approval rules still apply.

Publish a GitHub release for a canonical public version tag such as `v0.2.0`
or `v0.2.0rc1`. `cuda-release.yml` then:

1. Validates the version and builds Linux/Windows CPython 3.11–3.14 wheels using
   the shared workflow. A `v0.2.0` tag produces `0.2.0+cu129` CUDA metadata.
2. Requires all eight portable wheels, matching metadata, CPU/GPU modules,
   and no bundled NVIDIA driver.
3. Uploads the wheels to that existing release. It uploads `c-spikes-cu129.json`
   last, recording each filename, size, SHA-256 and Python requirement.
4. Calls `wheel-index.yml`, which reads all published release manifests and
   deploys a PEP 503 HTML index. Pages holds only small HTML files; downloads
   go directly to GitHub Release assets with `#sha256=` integrity fragments.

No PyPI project is needed for the CUDA variant, and no long-lived publishing
token is stored: release uploads use the job's `GITHUB_TOKEN`, while Pages uses
the Pages deployment action and OIDC. CUDA `+cu129` wheels must not be uploaded
to PyPI. CPU artifacts keep their public version without the CUDA suffix.

Incomplete releases are not indexed. Older versions and CUDA channels remain
available, and prerelease versions retain their prerelease version semantics.
Re-running the upload job accepts identical existing assets but refuses to
overwrite different bytes. Use **Re-run failed jobs** to reuse successful build
artifacts after a publication failure; rebuilding the same version can produce
different bytes and requires a new version rather than replacing a release.
The publishing workflow also accepts an existing release tag via manual dispatch.

Run **Deploy CUDA wheel index** manually to retry only Pages deployment or
reconcile deleted releases. It rebuilds from release assets without compiling
wheels. It refuses an empty deployment or a manifest with missing/corrupt assets.
Release upload and Pages deployment are separately retryable, so an unavailable
Pages configuration does not discard uploaded wheels.

For a local offline preview after generating release manifests:

```text
uv run --no-sync python scripts/wheel_index.py manifest --repository gerardjb/C-SPIKES --tag v0.2.0 --wheel-dir dist/cuda-release
uv run --no-sync python scripts/wheel_index.py build --repository gerardjb/C-SPIKES --manifests path/to/manifests --output build/wheel-index
python -m http.server --directory build/wheel-index
```

Save the first command's JSON output as `c-spikes-cu129.json` in the manifests
directory. The release job creates it automatically. A preview requires the full
release matrix and changes no remote state. The native GPU smoke/CTest checks
and scientific acceptance described above remain release validation tasks;
standard hosted runners check compilation and code images, not GPU execution.

The structure follows [Scientific Python cookie](https://github.com/scientific-python/cookie):
declarative packaging, an explicit supported matrix, isolated wheel tests, uv for
development/CI, and cibuildwheel for compiled extensions. See the upstream
[cibuildwheel options](https://cibuildwheel.pypa.io/en/stable/options/) and
[manylinux requirements](https://github.com/pypa/manylinux) for the platform boundary.
