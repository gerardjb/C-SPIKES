# Portable CUDA versus Hopper on H100 NVL

On 2026-11-01, a comparison on `della-rse` found no consistent end-to-end
performance advantage from building the current PGAS code specifically for
Hopper. Median inference times differed by less than 1% across the three cases.
Keep the portable release profile on this evidence.

The confirmation run used 30 MCMC iterations per inference, two warm-ups per
build per case, and six measured repetitions per build. These are wall-clock
seconds for `Analyzer.run()` plus a final CUDA synchronization:

| Case | Portable median (range) | Hopper median (range) | Hopper speedup |
|---|---:|---:|---:|
| 1,000 particles, default host resampling/full trajectory | 3.324 (3.305–3.336) | 3.313 (3.309–3.317) | +0.3% |
| 1,000 particles, device resampling/selected trajectory | 2.786 (2.772–2.804) | 2.812 (2.774–2.838) | −0.9% |
| 8,192 particles, device resampling/selected trajectory | 7.287 (7.187–7.478) | 7.325 (7.186–7.915) | −0.5% |

Speedup is `100 * (portable_median / hopper_median - 1)`; a negative value
means Hopper was slower. The initial six-iteration experiment also showed no
consistent advantage. Both experiments, including warm-ups and all outliers,
are retained in [the raw results](h100-cuda-architectures.json). Median paired
ratios are also recorded; those estimates differ by less than 1% from parity
in every case. These short measurements do not establish a universal 1% bound.

## Controls and validation

- Both wheels came from the same working-tree source archive, using CUDA SDK
  12.9.1 (nvcc 12.9.86), Kokkos 5.2.2, GCC 11.5, Python 3.14.7, NumPy 2.5.4 and
  identical pinned numerical dependencies. The archive and wheel hashes are
  recorded in the raw results. They are local Linux benchmark wheels.
- `PORTABLE` selected Kokkos `TURING75` and generated native `sm90` code from
  `compute_75`, alongside the other release targets. `HOPPER90` selected Kokkos
  `HOPPER90` and compiled `compute_90`/`sm90`. Both were Release builds.
- `CUDA_DISABLE_PTX_JIT=1` ensured both benchmarks executed native code. They
  ran in separate persistent Python processes with alternating AB/BA order,
  one active inference at a time, on GPU 0 (H100 NVL, driver 610.57.04).
- Host execution was pinned to CPU 2 on the GPU's NUMA node, with OpenMP and
  BLAS threads set to one. Sampled GPU SM clocks stayed at 1,785 MHz throughout
  the measured confirmation runs; temperatures were 44–54°C. Clocks and power
  settings were not changed. GPU 1 remained unused.
- Input was the first two seconds of trial 0 in
  `sample_data/janelia_8f/excitatory/jGCaMP8f_ANM478349_cell01.mat`:
  241 finite observations, converted to float64 without resampling or
  normalization. Both builds used the same constants, GCaMP parameters and
  seeds. Parameter sampling remained enabled.
- Imports, input loading and output checks were outside the timer. Native
  setup, parameter updates, transfers and trajectory file writes inside
  `Analyzer.run()` were included. Output files were on local `/tmp` storage.
- Each wheel passed 8 native checks, 91 native-core Python tests and 93 Python
  contract tests, plus CPU/GPU import-order smokes and forced-PTX GPU smokes.
  All benchmark parameter arrays and saved trajectories were byte-identical
  across builds and repetitions within each case and iteration count.

This measures application performance for a short trace, rather than isolated
kernel throughput or a converged scientific analysis. Different data, longer
traces, particle counts, future kernels or other GPU generations can behave
differently. The matching outputs do not replace historical numerical
qualification.

## Repeating the comparison

Use a Linux H100 machine with CUDA 12.9 and the native build prerequisites.
From the same source snapshot, prepare and validate two separate builds:

```bash
uv sync --managed-python --python 3.14.7 --locked --no-install-project
uv run --no-sync python scripts/build_native.py --backend cuda \
  --cuda-root /usr/local/cuda-12.9 --cuda-arch PORTABLE --jobs 4 --test \
  --build-dir build/bench-portable --output-dir dist/bench-portable
uv run --no-sync python scripts/build_native.py --backend cuda \
  --cuda-root /usr/local/cuda-12.9 --cuda-arch HOPPER90 --jobs 4 --test \
  --build-dir build/bench-hopper --output-dir dist/bench-hopper --skip-dependencies

uv export --locked --no-dev --no-emit-project --no-hashes -o build/bench-requirements.txt
for variant in portable hopper; do
  uv venv --python .venv/bin/python "build/bench-$variant-env"
  uv pip install --python "build/bench-$variant-env/bin/python" \
    -r build/bench-requirements.txt dist/bench-"$variant"/repaired/*.whl
done
```

Prepare the input from the sample data in the checkout (excluded from sdists):

```bash
uv run --no-sync python - <<'PY'
import numpy as np
from scipy.io import loadmat
x = loadmat("sample_data/janelia_8f/excitatory/jGCaMP8f_ANM478349_cell01.mat")
t = np.asarray(x["time_stamps"][0], dtype=float)
y = np.asarray(x["dff"][0], dtype=float)
keep = np.isfinite(t) & np.isfinite(y) & (t >= t[0]) & (t < t[0] + 2)
np.savez("build/bench-input.npz", time=t[keep], data=y[keep])
PY

CUDA_VISIBLE_DEVICES=0 taskset -c 2 .venv/bin/python scripts/benchmark_cuda_architectures.py \
  --portable-python build/bench-portable-env/bin/python \
  --hopper-python build/bench-hopper-env/bin/python \
  --input build/bench-input.npz \
  --constants parameter_files/constants_GCaMP8_soma.json \
  --gparams src/c_spikes/pgas/20230525_gold.dat \
  --output /tmp/c-spikes-architecture-benchmark \
  --iterations 30 --repeats 6 --warmups 2
```

Choose an idle GPU, a CPU on its NUMA node, and a new output directory. The
[harness](../../scripts/benchmark_cuda_architectures.py) records individual
timings, loaded extension hashes, output hashes, worker logs and summaries.
It sets the thread limits and disables PTX JIT itself.

The full local evidence bundle is
`build/h100-benchmark-results/evidence.tar.gz` (SHA-256
`630f3b52777e7fcd35b9e4bdaa358f9f07355ea68abca0078fbf0473f2777bf5`).
It retains both wheels, build/test logs, telemetry, configuration files, input,
saved trajectories and the exact harness. The remote workspace was
`/tmp/c-spikes-h100-dmturner.x9cUnR` on `della-rse`.
