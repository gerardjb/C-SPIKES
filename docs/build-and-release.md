# Linux, Windows and Apple Silicon builds

The release matrix targets CPython 3.11–3.14 on these platforms:

| Platform | Minimum OS / architecture | PGAS backends |
|---|---|---|
| Linux | manylinux_2_28, x86-64 | CPU OpenMP and CUDA |
| Linux | manylinux_2_28, aarch64 (ARM64) | CPU OpenMP and CUDA (SBSA servers) |
| Windows | x86-64 | CPU OpenMP and CUDA |
| macOS | 14.0 or newer, arm64 (Apple Silicon) | CPU OpenMP |

Every wheel includes CPU OASIS. CPU execution works without an NVIDIA driver.
CUDA execution on Linux (x86-64/ARM64) and Windows x86-64 is selected explicitly
and requires a compatible driver, but no system CUDA toolkit. Intel Macs, Windows ARM64, musllinux
and free-threaded Python remain outside this matrix.

`build_wheels.py` defaults to combined wheels on Linux (x86-64/ARM64) and Windows
x86-64, and CPU-only wheels on macOS ARM64. Source-checkout `uv sync` builds an
editable CPU package on every supported platform. See
[combined wheels](combined-wheels.md) for backend isolation.

## Build inputs

- scikit-build-core builds the existing CMake project. CMake defaults to CPU.
- uv manages Python environments using the committed `uv.lock`.
- `vcpkg.json` pins GSL 2.8, JsonCpp 1.9.6, serial LP64 OpenBLAS 0.3.29 and
  reference LAPACK 3.11.0. The triplets use CORE2 on x86-64 and ARMV8 on
  Linux/macOS ARM64, so distributable wheels do not depend on the build runner's CPU-specific instructions.
- The bootstrap pins vcpkg tooling separately from its package baseline. Windows
  explicitly enables vcpkg's MinGW Fortran provider and uses MSVC for C++. A small
  overlay port builds LLVM OpenMP 19.1.7 from hash-pinned upstream archives. This
  supplies Kokkos's required OpenMP API with `/openmp:llvm`, without bundling the
  nonredistributable experimental runtime from Visual Studio.
  MSVC's `_OPENMP` macro still reports 2.0, so CMake verifies the required LLVM
  runtime API and OpenMP linkage before reporting 3.0 to Kokkos's dependency check.
- Apple Silicon uses AppleClang and the same pinned LLVM OpenMP runtime, with
  GFortran 14 for reference LAPACK. macOS keeps OpenBLAS and LAPACK static, hides
  private symbols with a Mach-O export list, and uses delocate to bundle OpenMP
  and Fortran runtime dylibs and remove external library search paths.
- Kokkos [5.2.2](https://github.com/kokkos/kokkos/releases/tag/5.2.2) is pinned to
  commit `8e15454876298832f013ebcaf08d53825b5bda40` and uses C++20 for host and
  CUDA compilation. Its upstream Windows CUDA fixes replace the previous local
  Kokkos 4.3.01 backport; no Kokkos source patches are applied.
- CUDA compiler/runtime/CCCL and cuobjdump component archives are pinned by
  SHA-256 in `cmake/cuda-12.9.1.json` (nvcc 12.9.86), including Linux SBSA
  components for ARM servers. NVIDIA's `linux-aarch64` archives in CUDA 12.9
  target Jetson; these builds deliberately use `linux-sbsa` instead. Tools are unpacked under
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
Linux ARM64 uses the same commands and selects `arm64-linux-csp` automatically.
Kokkos 5 requires C++20; use GCC 10.4 or newer (the manylinux build uses GCC 12).
On Windows install Visual Studio 2022 C++ build tools and an x64 Windows SDK. Use
a native Windows checkout and PowerShell for Windows builds, and a Linux checkout
for WSL builds. Do not reuse build directories between operating systems, CPU
architectures, or compiler toolchains.

On Apple Silicon macOS 14 or newer, install Xcode command-line tools and Fortran:

```text
xcode-select --install
brew install gcc@14 pkgconf
```

The bootstrap discovers `gfortran-14`; an explicit `FC` setting can select another
compiler for local development. CI uses GFortran 14 on native `macos-14` runners.
Release builds target macOS 14.0, including the bundled library dependencies;
delocate rejects a library that needs a newer OS. Users of published wheels do
not need a compiler, Homebrew, or a separately installed OpenMP runtime.

From the checkout root, on any supported OS:

```text
uv sync
```

The checkout pins Python 3.14 for development. `uv sync` creates `.venv`, invokes
scikit-build-core/CMake, prepares the pinned native dependencies, and installs
the CPU package in editable mode. The first run needs network access and can
take several minutes to compile native dependencies. Later runs reuse the
dependency cache under `build/` and the editable build under
`build/editable/`. On Windows, the editable install includes its native runtime
DLLs beside the extension; no developer DLL search path is required. On macOS,
the editable build resolves its prepared runtimes under `build/` and the selected
Fortran compiler installation; distribution wheels use bundled runtime copies.

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
The wheel scripts use Ninja on all three operating systems. Plain `uv sync` uses
scikit-build-core's default generator (Visual Studio on Windows).

For CUDA on Linux/Windows, install a compatible NVIDIA display driver on the host, then:

```text
uv run --no-sync python scripts/prepare_cuda.py
uv run --no-sync python scripts/build_native.py --backend cuda --cuda-root build/tools/cuda-12.9.1 --cuda-arch PORTABLE --test
```

`PORTABLE` is the release profile, described below. For a separately tuned local
build, the script also accepts `TURING75`, `AMPERE80`, `AMPERE86`, `ADA89`,
`HOPPER90`, `BLACKWELL100`, `BLACKWELL103`, `BLACKWELL120` and `BLACKWELL121`.
Use a fresh build directory when changing the selected Kokkos architecture or
toolkit. Individual architecture builds are for local use, not publication to
PyPI. `prepare_cuda.py` does not install a display driver.
WSL uses the Windows host's driver.

CUDA `--test` requires a working GPU and fails if the GPU extension or its kernels
cannot run. The independent CPU extension remains usable without CUDA or an NVIDIA driver.
CUDA validation also exercises both backends in one process in both import
orders and checks CPU execution after an unavailable-GPU request.

Install the single repaired wheel from the chosen build's `repaired/` directory
using `uv pip install /path/to/the-wheel.whl`. Combined wheels use the ordinary
`c-spikes` version and the `c_spikes` import package. CPU-only local builds use
the same distribution name; install one wheel per environment.
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

## Linux ARM64 validation

[CI run 41673321821](https://github.com/gerardjb/C-SPIKES/actions/runs/41673321821)
built the ARM implementation at commit `edb7a27eacba70408c8c2c67440ef44e736b8b26`.
Its native Ubuntu ARM64 job passed `uv sync`, both import-order smoke checks,
all five C++ tests, and 91 native-core plus 93 Python contract tests on the
repaired Python 3.14 wheel. Kokkos uses OpenMP, with GCC/GFortran 13.3.0.
The downloaded development wheel's extensions and bundled OpenMP/Fortran
runtimes are AArch64. Its Ubuntu glibc 2.39 tag is for local validation; the
release wheels come from the separate manylinux_2_28 job.

The ARM manylinux job built combined CPU/CUDA wheels for Python 3.11–3.14 using
GCC/GFortran 12.2.1 and nvcc 12.9.86. Every repaired wheel passed all 184 Python
tests, both NumPy import orders, and CUDA-unavailable/CPU-reuse checks. cuobjdump
verified all nine native GPU targets and `compute_75` PTX in each GPU extension,
and no device code in the CPU extension.

All four downloaded wheels carry `manylinux_2_27_aarch64` and
`manylinux_2_28_aarch64` tags. Inspection confirmed five AArch64 ELF binaries per
wheel: CPU PGAS, CUDA PGAS, OASIS, GFortran and OpenMP. Search paths are relative
to the package; dependencies are bundled or manylinux system libraries, with
the NVIDIA driver required only by the GPU extension and excluded from bundling.
The 65 packaging checks pass, and the sdist includes the ARM triplet and pinned
SBSA provisioning inputs. Artifacts and an ELF inspection receipt are retained
under `build/linux-arm64-ci-validation/` locally.

ARM CUDA GPU execution remains pending hardware access. These results qualify
the builds and CPU operation, not execution or performance on Grace/Blackwell,
Jetson support, optional ML/GUI frameworks, or historical numerical references.

## Apple Silicon validation

[CI run 41561755714](https://github.com/gerardjb/C-SPIKES/actions/runs/41561755714)
built the macOS implementation at commit `9bc98c720c43fc00e26f146743feeb2aab8c258d`.
The native `macos-14` job passed the editable `uv sync` import checks, all five
native C++ checks, and 91 native-core plus 93 Python contract tests on the repaired
Python 3.14 wheel. Kokkos reports OpenMP as its host parallel execution space.
The separate wheel job passed the same 184 Python tests on each of Python
3.11, 3.12, 3.13 and 3.14, plus both NumPy import orders and clean CUDA-unavailable
behavior followed by CPU inference.

The four `macosx_14_0_arm64` wheels were downloaded and inspected: both extensions
and all four bundled runtime dylibs are ARM64. Their dependencies refer only to
package-relative locations or macOS system libraries; there are no external
rpaths. OpenMP, GFortran, quadmath and GCC support libraries are bundled. Local
copies of the artifacts and inspection receipt are under `build/macos-ci-validation/`.
This is build/runtime qualification on macOS 14.8.9, not the handoff's historical
numerical-reference acceptance or qualification of optional ML/GUI frameworks.

## Local validation

The current combined-wheel results are recorded in [combined wheels](combined-wheels.md):
Windows and Linux CPython 3.14 passed eight native checks, 184 Python tests,
CPU/GPU coexistence, and GPU native/forced-PTX execution. Linux additionally
passed CPU tests and consumer `uv sync` in a container without an NVIDIA driver.
The following earlier runs document the portable CUDA architecture work before
the independent CPU/CUDA packaging change. They are not full-matrix qualification
of the combined wheel.

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

The Python regression profiles exercise CPU; GPU execution is covered by the
native checks and Analyzer smoke runs. The earlier GitHub Pages wheel-index
implementation and its validation are preserved at `eb4c7a2` on
`build/cuda-wheel-index`; that publishing path is superseded by PyPI here.

## CI and release artifacts

`.github/workflows/build.yml` runs on pull requests, main/master pushes and
manual dispatch. `release.yml` also calls it for a version-tag release. It
builds/tests CPU development installs and native C++ checks on Ubuntu x86-64 and
ARM64, Windows x86-64 and Apple Silicon with Python 3.14. It creates an sdist,
tests release validation and platform selection, and builds Python 3.11–3.14
wheels for all four platform/architecture combinations.
Linux (x86-64/ARM64) and Windows x86-64 wheels contain both backends; macOS
wheels contain CPU only.

Every wheel is repaired and runs installed CPU/import/regression tests
on hosted runners, including lazy loading and unavailable-CUDA behavior. CUDA
execution still requires the separate hardware checks; hosted builds inspect the
GPU code images without executing kernels.

Linux ARM64 jobs use native `ubuntu-24.04-arm` runners for both source installs and
all four wheel ABIs; no QEMU emulation is used. ARM wheels target baseline ARMv8
and glibc 2.28 or newer, with auditwheel bundling the required OpenMP and Fortran
runtimes. These builds cover 64-bit ARM Linux, not 32-bit ARM or musl-based Linux.
ARM CUDA builds use the pinned SBSA toolkit and include Hopper (`sm_90`),
Blackwell (`sm_100`/`sm_103`) and DGX Spark (`sm_121`) code alongside the other
portable images. They are intended for ARM servers such as Grace Hopper and
Grace Blackwell. Hosted CI inspects those images and tests CPU fallback without
a GPU; execution on ARM GPU hardware remains unqualified. Jetson/JetPack is not
covered by this SBSA build. Rubin is not an explicitly supported GPU target in
CUDA 12.9; native Rubin support requires a separate toolkit upgrade and validation.
NVIDIA introduced the native Rubin target as a preview in
[CUDA 13.4](https://developer.nvidia.com/blog/cuda-toolkit-13-4-adds-windows-on-arm-support-and-greater-control-over-shared-gpus/).

The manylinux jobs use the `manylinux_2_28` image selected by the pinned cibuildwheel
version and GCC toolset 12, including Fortran. Linux wheels built directly on
Ubuntu/WSL have that system's glibc baseline; they are local validation artifacts,
not replacements for the manylinux job. Auditwheel must succeed for portable
Linux distribution. NVIDIA's `libcuda.so.1` and `nvcuda.dll` drivers are explicitly
excluded from bundling, including WSL's host-driver proxy. Native dependency and CUDA runtime
notices are included in the wheels.

Windows repair uses `scripts/repair_windows_wheel.py` to keep delvewheel from
writing renamed DLL imports into CUDA section padding. DLL bundling and name
mangling remain enabled; the repaired GPU module must still pass cuobjdump's
native-image and PTX checks.

To reproduce cibuildwheel locally (Linux requires Docker or Podman):

```text
uv sync --locked --no-install-project --group ci
uv run --no-sync python scripts/build_wheels.py --output-dir dist/wheels
# Optional CPU-only development artifacts:
uv run --no-sync python scripts/build_wheels.py --backend cpu --output-dir dist/wheels-cpu
```

CUDA cibuildwheel builds default to `--cuda-arch PORTABLE`. Every repaired CUDA
wheel is inspected with cuobjdump on the build runner, checking all native images
and PTX in the GPU module and requiring no device code in the CPU module.

`--platform linux` also permits running the Linux container builds from Windows
with Docker Desktop. `--print-build-identifiers` previews the selected matrix.
To qualify one existing CUDA wheel on a GPU machine, place it alone in a directory
and run `python scripts/test_wheel.py --wheel-dir <directory> --backend cuda --skip-repair --cuda-root build/tools/cuda-12.9.1`
from the prepared uv environment. This tests the exact repaired artifact.

`.github/workflows/cuda-hardware.yml` is manual-only and requires an explicitly
registered self-hosted runner labeled `c-spikes-cuda`, plus the matching OS and CPU architecture
labels (`x64` or Linux `ARM64`). Select the host architecture separately from the
GPU architecture when dispatching it. It runs the same CTest and installed-wheel CUDA execution checks as the
local script, including forced-PTX smoke checks. `PORTABLE` is its default profile.
No runner registration is performed by this change.

`wheels.yml` is reused by both CI and tagged releases. The selected artifacts are
`wheels-Linux-X64-cuda`, `wheels-Windows-X64-cuda`, `wheels-Linux-ARM64-cuda`,
`wheels-macOS-ARM64-cpu` and `sdist`. Only these are admitted to PyPI; native
development wheels are never selected for publishing.

## CUDA GPU coverage

The portable wheel uses CUDA 12.9.1 and the Kokkos Turing feature baseline.
Kokkos 5.2.2 permits one `Kokkos_ARCH_*` setting. All device code is therefore
compiled as `compute_75`, with native machine code generated for each target:

| GPU generation | Native code images | Runtime validation |
|---|---|---|
| Turing, including RTX 2080 Ti | sm75 | Local Linux/WSL and Windows |
| Ampere | sm80, sm86 | Compiled; hardware testing needed |
| Ada | sm89 | Compiled; hardware testing needed |
| Hopper | sm90 | Compiled; hardware qualification outside this PR |
| Blackwell | sm100, sm103, sm120, sm121 | Compiled; hardware testing needed |

The wheel also retains `compute_75` PTX for the driver's JIT compiler. We avoid
architecture-specific `a`/`f` targets, which have narrower compatibility rules.
The GPU PGAS module and its Kokkos runtime carry these code images.
The independent CPU module contains no CUDA device code. This uses standard
nvcc code generation and does not patch Kokkos or attempt to enable several
Kokkos architecture configurations together. See NVIDIA's
[compiler documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html)
and [Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.2/blackwell-compatibility-guide/index.html).

The common feature baseline preserves the 2080 Ti path. It does not enable every
newer GPU optimization; Kokkos can warn that its compiled feature baseline differs
from the detected device. Newer GPUs still need hardware qualification and
performance measurements. Code images do not extend the Python wheel's host
platform support: ARM systems install a separate aarch64 wheel containing ARM
host code, even when its GPU code images match the x86-64 wheel's images.

Installed-wheel CUDA tests run real Analyzer kernels normally and again with
`CUDA_FORCE_PTX_JIT=1` and the JIT cache disabled. The latter verifies the shipped
PTX path on available hardware; it does not substitute for testing a Blackwell
device. The GPU extension statically links cudart, so GPU users need an NVIDIA
driver but no CUDA toolkit or separate NVIDIA runtime pip packages. CPU users
need neither the toolkit nor the driver. Optional
PyTorch/TensorFlow integrations have their own runtime requirements.

Use a driver supporting CUDA 12.9 Update 1 or newer. NVIDIA lists Linux
575.57.08 and Windows 576.57 for that toolkit release; these are conservative
installation guidance, not locally tested minimums. Older CUDA 12-compatible
drivers may run native images but cannot be assumed to JIT this toolkit's PTX.
See [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html).

## Installing published wheels

After the first release containing this packaging is published, the same command
installs CPU OpenMP on every supported platform, plus CUDA on Linux/Windows:

```text
uv pip install c-spikes
```

Use `uv pip install --upgrade c-spikes` to update, or pin a published version with
`uv pip install "c-spikes==0.2.0"` (`0.2.0` is an example, not a claimed release).
For a consumer uv project:

```text
uv add c-spikes
uv sync
```

Commit the generated `pyproject.toml` and `uv.lock`. Subsequent syncs use the lock;
`uv sync --upgrade-package c-spikes` selects a newer compatible release. No
custom index or source mapping is required. `uv sync` in this repository is a
source development install and continues to build CPU by default.

CPU is the runtime default. Set `C_SPIKES_PGAS_BACKEND=cuda` for the inference
pipeline, or select the native backend after import:

```python
from c_spikes.pgas import get_backend
cpu = get_backend("cpu")
gpu = get_backend("cuda")  # Linux/Windows with a compatible NVIDIA GPU
```

The modules can coexist; existing Analyzer objects remain on their original
backend. An explicit CUDA request fails with an actionable error when the driver
or GPU is unavailable. See [combined wheels](combined-wheels.md).

## Publishing tagged releases to PyPI

One-time setup by the repository/PyPI owner:

1. Configure a PyPI Trusted Publisher for project **c-spikes**, repository owner
   **gerardjb**, repository **C-SPIKES**, workflow **release.yml**, environment
   **pypi**. A pending publisher can create the project on its first upload.
2. Configure the GitHub **pypi** environment to permit **tags matching `v*`**.
   A main-branch-only environment rule blocks this tag-triggered workflow.
   If environment reviewers are configured, their approval is still required.
3. Merge these workflows into `main` before tagging a commit that contains them.
   This workflow replaces the manual backend-selection design in PR #14;
   reconcile that PR before merging both implementations. No
   `PYPI_PUBLISHING_ENABLED` variable or long-lived API token is used here.

Publishing begins when a canonical version tag such as `v0.2.0` or `v0.2.0rc1`
is pushed. Creating a GitHub Release is optional and does not trigger a second
upload. A pull request or ordinary branch push cannot publish.

`release.yml` performs these steps:

1. Validate the tag (no development/local suffix) and require the tagged commit
   to belong to `main` history.
2. Run the same full build/test workflow used by PRs, producing twelve combined
   Linux x86-64/ARM64 and Windows x86-64 wheels, four CPU-only macOS ARM64
   wheels and one sdist. All build jobs must succeed before preparing a release.
3. Download the five exact artifact names from that same workflow run. Validate
   all sixteen CPython/platform/architecture slots, public version and metadata,
   the required PGAS backends for each platform, the lazy loader, and absence of NVIDIA drivers.
   macOS requires the `macosx_14_0_arm64` tag and CPU only. Linux requires
   manylinux_2_28 or older, CPU+CUDA on both aarch64 and x86-64; local
   Ubuntu wheels and mixed-architecture platform tags are rejected.
4. Record the source SHA, run identity and each distribution's SHA-256 in
   `release-manifest.json`, then upload the validated set as `pypi-distributions`.
5. In a separate job using the `pypi` environment, verify those hashes and publish
   with PyPI Trusted Publishing and attestations. Only this job receives
   `id-token: write`; it does not check out or build package code.

A `v0.2.0` tag produces version `0.2.0` for every platform, with CPU OpenMP on
macOS ARM64 and both CPU/CUDA backends on Linux x86-64/ARM64 and Windows x86-64.
Prerelease tags retain prerelease semantics on PyPI. The publisher refuses
existing filenames rather than silently skipping a possibly different build.
After a failure before any upload, use **Re-run failed jobs** to reuse successful
artifacts. If an upload partially succeeds, inspect PyPI before retrying; do not
move the tag or rebuild and replace already published files. A new version may
be needed to publish a corrected complete release.

The local bundle check does not upload:

```text
uv run --no-sync python scripts/validate_release.py bundle --tag v0.2.0 --directory dist/release --manifest build/release-manifest.json
```

Native GPU qualification and scientific acceptance remain release-owner tasks;
hosted CI checks CPU behavior and GPU code images. The workflow does not claim
that the historical numerical-reference suite has passed. See PyPI's
[Trusted Publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

The structure follows [Scientific Python cookie](https://github.com/scientific-python/cookie):
declarative packaging, an explicit supported matrix, isolated wheel tests, uv for
development/CI, and cibuildwheel for compiled extensions. See the upstream
[cibuildwheel options](https://cibuildwheel.pypa.io/en/stable/options/) and
[manylinux requirements](https://github.com/pypa/manylinux) for the platform boundary.
