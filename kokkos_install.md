# Native build prerequisites

Start with the [Linux numerical-provider requirements](docs/numerical-provider-linux.md).
The current provider profile requires the matching hashed OpenBLAS/LAPACK archives
and compatible GSL/CBLAS and runtime dependencies. Installing arbitrary versions
with a package manager does not satisfy that contract. The production module no
longer requires Armadillo; some opt-in reference tests still require its headers.

The repository's `pyproject.toml` contains a vcpkg toolchain path and CUDA/AMPERE80
build choices. These are explicit build inputs to review for your environment,
not a self-contained dependency installer or a universal GPU recipe. For the
recorded CUDA toolchain, compiler flags and architecture limits, see
[CUDA build notes](docs/cuda-build-notes.md). Verify the emitted compiler commands,
including the CUDA constexpr flag described there.

After preparing the compatible dependencies and CMake configuration, an editable
full-package build can be requested with:

```bash
python -m pip install -ve .
# Include the test dependencies when needed:
python -m pip install -ve '.[test]'
```

These commands also resolve the full Python package's dependencies; they are not
a minimal native-core installer. For the separate CPU PGAS/OASIS install component,
use [native-core installation](docs/native-core-installation.md). See
[test profiles](tests/README.md) before choosing which tests to run. A working
native build does not establish training, optional-backend or wheel portability.
