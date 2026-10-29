# GPU PGAS validation

This page describes focused contributor checks and historical measurements for
the [optional GPU PGAS modes](pgas_gpu_options.md). Passing a small sampler or
extraction fixture does not establish full inference accuracy, posterior
equivalence, general performance or support for every GPU/toolchain.

## Build prerequisites

Use separate CPU and GPU build directories with a working native-extension
configuration. The commands below add test targets to an existing configuration;
they are not a complete dependency-installation or clean-build recipe. Project
name/version, Python, NumPy, pybind11, GSL, JsonCpp and the numerical-library and
Kokkos dependencies must already be configured for this source revision. Consult
the [installation guide](../README.md) and [dependency notes](../kokkos_install.md),
and verify their assumptions against your environment before building.

For GPU tests, use CUDA-enabled Kokkos, a compatible host compiler/CUDA toolkit
and driver, and an architecture setting appropriate to the actual device. Do not
assume the historical versions below are universal requirements or that a different
GPU is qualified. A CUDA-enabled Kokkos initialization needs GPU access even when
the selected test uses OpenMP; use a genuinely CPU-only Kokkos configuration for
CPU-only validation. Run GPU checks only with an appropriate GPU allocation.

## Focused native and Python checks

From the repository root, with `build-cpu` and `build-gpu` already configured:

```bash
# CPU-only Kokkos configuration.
cmake -S . -B build-cpu -DPGAS_BUILD_GPU=OFF -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_OPENMP=ON -DC_SPIKES_TEST_RESAMPLING=ON
cmake --build build-cpu --target test_ancestor_cpu test_selected_cpu
OMP_NUM_THREADS=1 ctest --test-dir build-cpu -R '^(ancestor|selected)_cpu$' --output-on-failure

# Preserve the admitted CUDA toolchain and actual-device architecture settings.
cmake -S . -B build-gpu -DPGAS_BUILD_GPU=ON -DKokkos_ENABLE_CUDA=ON \
  -DC_SPIKES_TEST_RESAMPLING=ON
cmake --build build-gpu --target test_ancestor_gpu test_selected_gpu
OMP_NUM_THREADS=1 ctest --test-dir build-gpu -R '^(ancestor|selected)_gpu$' --output-on-failure

# Requires the Python test dependencies and importable package/native extensions.
PYTHONPATH=src python -m pytest -q tests/test_resampling_options.py
```

`C_SPIKES_TEST_RESAMPLING` defaults to OFF. Configure it explicitly before building
these targets. The native tests cover Philox known answers, GSL uniform consumption,
CDF endpoints and zero/extreme weights, invalid weights, conditional routing,
counter ownership and 200,000 paired categorical draws. CPU tests exercise the
portable sampler; GPU tests additionally exercise its device specialization.
Selected-trajectory fixtures compare histories and CPU fluorescence for small
particle/time grids and every terminal particle. Python checks cover option
validation and cache isolation. These tests serve different contracts; a Python
options pass is not evidence that device kernels ran.

## Historical measurements: October 2026

The original optional-mode validation used Kokkos 4.3.01, GSL 2.8, GCC 11.5,
CUDA 12.9, double precision and an A100 PCIe 3g.40gb MIG slice (42 SMs).
The default CPU path matched its reference raw outputs on a small parity fixture.
These are historical observations for that build and hardware, not a claim that
all current builds or supported-looking configurations have been revalidated.

| Separate comparison | Workload | Reported result |
|---|---|---|
| Device ancestor resampling | Two 2,001-frame epochs, 1,000 particles, 200 total sweeps including 100 burn-in; twelve fits | 8.17% lower pooled wall time |
| Selected trajectory extraction | A separate four-fit comparison | 5.25% lower wall time, with identical paired raw outputs |

Do not multiply these reductions into an unmeasured combined speedup. Two epochs
do not establish broad posterior equivalence or general performance. Large count
biases were still present. Turing hardware and ordinary-fit peak GPU memory or
utilization were not evaluated in those measurements. Keep these modes opt-in
until validation supports your intended data, settings and hardware. Validation
of the default host/full path alone does not validate the optional modes.
