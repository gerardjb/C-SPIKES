# GPU PGAS options

GPU PGAS offers two independent, opt-in operations: sampling particle ancestors
on the GPU and downloading only the selected trajectory. The defaults remain
host ancestor sampling and full-history extraction. Use the defaults unless you
intend to evaluate the optional modes for your workload; their focused tests and
limited historical comparisons do not establish broad posterior equivalence or
a general speedup. See [validation and historical results](pgas_gpu_validation.md).

## Runtime controls

Set these variables before importing the backend and constructing an inference
object. Backend selection occurs at import; native SMC construction reads the
sampling and extraction options. Start a fresh process when changing backends.

| Environment variable | Default | Effect |
|---|---|---|
| `C_SPIKES_PGAS_BACKEND` | GPU if importable, otherwise CPU | `cpu` or `gpu` explicitly selects a backend; an explicit unavailable backend raises an import error. |
| `C_SPIKES_PGAS_RESAMPLING` | `host` | `device` samples multinomial ancestors on the GPU. |
| `C_SPIKES_PGAS_ANCESTOR_SEED` | Effective SMC CPU seed | Optional decimal unsigned 64-bit key, accepted only with device resampling. |
| `C_SPIKES_PGAS_TRAJECTORY` | `full` | `selected` backtraces on the GPU and downloads the selected trajectory. |

Device resampling and selected extraction can be enabled separately or together
on a GPU build. Invalid values fail explicitly; CPU native builds reject either
device operation. To return to CPU inference, unset the three optimization
variables before selecting `cpu` in a fresh process.

These controls do not change the configured likelihood, conditional ancestor
weights, priors, noise calibration, particle count, iteration count or burn-in.
They can change sampled paths and floating-point accumulation order, so matching
settings and seeds do not imply identical output across sampling modes.

## Seeds and reproducibility

Device ancestors use Philox4x32-10 in a separate ancestor-operation domain.
Counters include the particle, timestep and both halves of the SMC-owned 64-bit
sweep index. Record a distinct ancestor seed for each fit or chain. Reusing the
same key and counters reproduces ancestor uniforms; worker identity and launch
order do not add entropy. Without an explicit key, the effective CPU seed is used,
including the existing constants-file fallback.

Particle 0 uses conditional ancestor weights; other particles sample ordinary
weights independently, with replacement. Device log weights are stabilized and
converted to cumulative distribution functions. Negative infinity represents zero
mass. NaN, positive infinity and all-zero mass cause an error at the sweep boundary;
there is no uniform fallback. Device CDF sampling and host GSL alias sampling need
not choose the same ancestors from the same seed.

With device resampling, the host GSL generator still consumes the original number
of ancestor uniforms before terminal selection and parameter updates. The separate
Kokkos movement pool resets to seed 42 each sweep. Changing the ancestor seed does
not independently reseed those movement proposals. Record both seed policies when
comparing fits; the ancestor seed alone is not a complete RNG description.

Selected extraction retains the CPU/GSL terminal draw and CPU fluorescence
calculation. It transfers all twelve selected calcium-state components, baseline,
burst and spikes. Initialization still allocates and uploads full histories, so
selected extraction does not by itself remove full-history GPU memory use.

## Cached results and external launchers

The Python inference adapter records sampling and extraction modes, RNG version
and seed policy. Optional modes have distinct cache identities and disable legacy
cache fallback. Default host/full lookup rejects optional-mode entries even when
older configuration keys are compared; default host/full cache identity remains
compatible.

If you use an external launcher or completion database, include the modes and
effective seeds in its task identity too. Changing environment variables alone
does not make an existing outer completion record safe to reuse.

For contributor build checks and the limits of the available performance evidence,
see [GPU PGAS validation](pgas_gpu_validation.md).
