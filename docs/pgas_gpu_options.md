# GPU PGAS options and developer checks

## Runtime controls

Set options before starting inference; native SMC construction reads them once.
Start a fresh process when switching the native backend.

| Environment variable | Default | Effect |
|---|---|---|
| `C_SPIKES_PGAS_BACKEND` | GPU if importable, otherwise CPU | Existing backend selector; `cpu` or `gpu` forces a backend. |
| `C_SPIKES_PGAS_RESAMPLING` | `host` | `device` samples multinomial ancestors on GPU. |
| `C_SPIKES_PGAS_ANCESTOR_SEED` | Effective SMC CPU seed | Optional decimal unsigned 64-bit key; allowed only with device resampling. |
| `C_SPIKES_PGAS_TRAJECTORY` | `full` | `selected` backtraces on GPU and downloads the selected trajectory. |

The two optimization controls are independent on GPU. Host/full and device/full
remain available as references. Invalid values fail explicitly; requesting either
device operation on a CPU native build is rejected. To restore CPU inference,
unset the three optimization variables before selecting `cpu`. Likelihoods,
conditional ancestor weights, priors, noise calibration, particles, sweeps and
burn-in are unchanged by these switches.

## RNG and numerical behavior

Device ancestors use Philox4x32-10 with a separate ancestor-operation domain.
Counters contain particle, timestep and both halves of the SMC-owned 64-bit
sweep index. Choose a distinct recorded ancestor seed for each fit/chain. Reusing
the same key and counters reproduces ancestor uniforms; worker identity and
launch order add no entropy. Without an explicit key, the effective CPU seed
supplies it, including the existing constants-file fallback.

Particle 0 samples conditional ancestor weights; other particles sample ordinary
weights independently, with replacement. Log weights are stabilized on device and
converted to CDFs. Negative infinity denotes zero mass. NaN, positive infinity
and all-zero mass raise an error at the sweep boundary; there is no uniform
fallback. CDF and GSL alias sampling need not produce the same seeded trajectory;
floating-point accumulation order can also differ.

The old number of ancestor GSL uniforms is consumed at sweep boundaries to
preserve its position before terminal selection and parameter updates. The
existing Kokkos movement pool still resets to literal seed 42 each sweep. These
options do not repair that lifecycle or independently reseed GPU proposals.
No existing named synchronization fence is removed.

Selected extraction retains the CPU/GSL terminal draw and CPU fluorescence
calculation. It transfers all twelve selected calcium-state components, baseline,
burst and spikes. Initialization still allocates and uploads full histories;
this option does not redesign particle storage.

## Cache and provenance

The inference adapter records resampling/extraction mode, RNG version and seed
policy. Experimental modes add this provenance to cache identity and disable
legacy fallback. Legacy host lookup rejects experimental entries even when
comparing older configuration keys. Host/full cache identity remains compatible.
External launchers must include modes and effective seeds in their own task and
completion identity; changing only environment variables cannot safely reuse an
outer completion record.

## Build and test

Use the normal dependency setup in [kokkos_install.md](../kokkos_install.md).
Enable `C_SPIKES_TEST_RESAMPLING=ON` for native tests. Direct CMake builds also
need the repository's normal project, Python, dependency and Kokkos configuration.
Assuming `build-cpu` and `build-gpu` are already configured with those settings,
apply these additions in separate build directories:

```bash
# A CUDA-enabled Kokkos initialization needs a GPU, even for OpenMP tests.
cmake -S . -B build-cpu -DPGAS_BUILD_GPU=OFF -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_OPENMP=ON -DC_SPIKES_TEST_RESAMPLING=ON
cmake --build build-cpu --target test_ancestor_cpu test_selected_cpu
OMP_NUM_THREADS=1 ctest --test-dir build-cpu -R '^(ancestor|selected)_cpu$' --output-on-failure

# Supply the architecture and CUDA toolchain for the actual device.
cmake -S . -B build-gpu -DPGAS_BUILD_GPU=ON -DKokkos_ENABLE_CUDA=ON \
  -DC_SPIKES_TEST_RESAMPLING=ON
cmake --build build-gpu --target test_ancestor_gpu test_selected_gpu
OMP_NUM_THREADS=1 ctest --test-dir build-gpu -R '^(ancestor|selected)_gpu$' --output-on-failure

PYTHONPATH=src python -m pytest -q tests/test_resampling_options.py
```

Run GPU checks in an appropriate allocation. Configure explicitly before building
new targets. CPU checks use the same portable sampler; GPU checks exercise the
device specialization. Tests cover Philox known answers, linked GSL consumption,
CDF endpoints/zero/extreme weights, invalid weights, conditional routing, counter
ownership and 200,000 paired categorical draws. Extraction fixtures compare exact
histories and CPU fluorescence for tiny N/T cases and every terminal particle.
Python checks cover options and cache isolation.

## Tested scope and limits

Validation used Kokkos 4.3.01, GSL 2.8, GCC 11.5, CUDA 12.9, double precision and
an A100 PCIe 3g.40gb MIG slice (42 SMs). Default CPU raw outputs matched main on a
small parity fixture. Two 2,001-frame epochs with 1,000 particles and 200 sweeps
(burn-in 100) showed an 8.17% pooled wall-time reduction for device resampling
across twelve fits. A separate four-fit comparison showed another 5.25% reduction
for selected extraction, with identical paired raw outputs. These are separate
comparisons; do not multiply their gains into an unmeasured benchmark.

Two epochs cannot establish broad posterior equivalence or general performance.
Existing large count biases remain. No Turing hardware was tested, and ordinary-fit
peak GPU memory/utilization was not measured. Keep both paths opt-in pending
broader validation. Campaign records and job-specific scripts are archived outside
the product tree; the PR descriptions identify the development-records archive.
