# CUDA build notes

This page records the historical Kokkos 4.3.01 validation. Current portable
builds use Kokkos 5.2.2 and C++20; see the [build and release guide](build-and-release.md).
Their CMake configuration enables `--expt-relaxed-constexpr` for CUDA and connects
the native fixtures to CTest with `C_SPIKES_TEST_RESAMPLING=ON`.

The native PGAS CUDA extension uses Kokkos execution and memory spaces. Its
terminal-weight copy uses a named internal device-callable functor so it can be
launched from a private method without CUDA extended-lambda access restrictions.
The functor copies view handles and assigns the selected history column; it does
not change ownership, the range policy or the subsequent host-copy boundary.
The Python API and native source/rebuild interface are unchanged by this repair.

## Recorded working configuration

The tested Linux configuration used Release C++17, Kokkos 4.3.01, CUDA toolkit
12.9.1 / nvcc 12.9.86, GNU 11.5, and Kokkos AMPERE80, with one build/runtime CPU
thread. It included `--expt-relaxed-constexpr` in the nvcc compile flags. Keep that
flag explicit for this recipe and verify the emitted compiler commands and
warnings. It is not currently enabled by a project CMake or TOML default.
Do not add it to ordinary CPU compiler flags.

The flag permits the host/device constexpr usage required by this tested
compiler/header combination. The recorded build without it emitted unsupported
host/device constexpr warnings and its runtime stalled; those observations do
not establish a universal requirement for all CUDA or Kokkos releases. A different
compiler/toolkit combination needs its own build and runtime checks. The named
functor addresses the separate private-method lambda compilation restriction.

AMPERE80 identifies the compiled device architecture, not a universal GPU binary.
Available architecture selectors do not establish tested support for T4, V100,
P100, future GPUs, macOS or Windows. Match the Kokkos architecture, toolkit, driver
and target hardware explicitly. Runtime libraries and numerical-provider ownership
must also be admitted; see the numerical-provider documentation.

## Direct copy and smoke checks

`tests/native/terminal_weight_copy.cpp` is a separate native fixture. Build it
against the same admitted Kokkos headers/libraries and the PGAS include directory.
Its CPU mode uses Kokkos OpenMP; define `CSP_COPY_TEST_CUDA` for its CUDA mode, with
CUDA enabled in that Kokkos build. The fixture is not automatically built by the
normal project or collected by pytest. Its build command must be reviewed for the
chosen toolchain rather than inferred from a passing CPU extension build.

Each execution space checks nine combinations of range length and selected column,
including zero work, signed zero, untouched output tails and unchanged input
history. It retains the production host-visibility boundary without adding a
fence to hide ordering errors. CPU success is not GPU qualification.

Keep the ancestor and selected-trajectory fixtures: they protect different
sampling and trajectory-copy contracts and are not superseded by this small copy
check. A full CUDA extension build, direct CPU/CUDA fixture, installed/import and
provider checks, and bounded one-/two-sweep smoke establish different properties.
Tiny smoke outputs do not establish full-epoch accuracy or performance, and a
successful in-tree CUDA build does not establish a GPU wheel installation.
