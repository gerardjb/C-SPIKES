# Native build prerequisites

Use the [build and release guide](docs/build-and-release.md) for the reproducible
Linux/Windows CPU and CUDA wheel builds. It covers uv, the pinned vcpkg dependency
manifest, CUDA provisioning, installed-wheel tests and CI artifacts. CMake fetches
Kokkos 5.2.2 at a fixed commit and builds with C++20; a separate global Kokkos
installation is unnecessary.

The [historical Linux numerical-provider profile](docs/numerical-provider-linux.md)
still requires the original archive/runtime bundle. Selecting it does not restore
the handoff's older Kokkos version or reproduce its qualified environment.
The production module no longer requires Armadillo; some optional reference tests
still require its headers.

See [test profiles](tests/README.md) before choosing additional qualification
tests. Optional training frameworks and GUI operation need their own validation.
