# Opt-in checked CPU sweep preparation

`SMC::try_prepare_sweep(const param&, const checked::Context&)` is enabled only by OFF-by-default `C_SPIKES_TEST_SWEEP_CPU`, requiring CPU-only `C_SPIKES_TEST_CHECKED_CPU`. It returns a checked report and optional owned GCaMP_params snapshot. It does not start a sweep and is not automatically used by PGAS, sampleParameters or Analyzer. No Python binding is added.

Precondition: an existing valid SMC with initialized model from supported data/time construction, and exclusive ownership of the GSL instance in an isolated process. The method inherits the checked API's restricted RAII context and numerical-failure semantics; it is not a general embedded-host API. Missing/busy contexts produce status before backend/handler changes. A model-less or otherwise invalid legacy SMC is not made valid by this method.

Success atomically applies six cell parameters and initialization using the existing checked composite. State/initial state, normalization, final caches and DFF reset follow the checked composite-update contract; histories and Cparams retain values. Object identity is stable. Reacquire borrowed native model storage after success. The handoff is constructed in fixed scalar/array storage and read_params fills it only after successful commit; no heap allocation or caller mutation occurs during handoff construction. It includes named fields and fixed constants, not model storage views or a TSMode selector.

Numerical failure returns diagnostics and no handoff, preserving model contents and storage references. Never treat an absent handoff as valid initialization. There are no new numerical exceptions, retries, implicit MH rejection, warnings, integrator changes or default dispatch changes. Existing resource/type exception behavior is not redefined.

On both paths RNG state, sweep counter, particle system, caller parameters/configuration, trajectories and files remain untouched. This is one preparation boundary, not whole-sweep/proposal/run rollback. No sampling or Euler trajectory qualification is implied; the1uM fixture is initialization only. New calls do not grant permission to resume a failed full run.

Focused CPU fixtures observe only initialized named fields and explicitly seed
otherwise indeterminate values; they do not compare raw object padding. See the
[fixture notes](../tests/native/contracts/README.md). A successful preparation is
not evidence of a valid full sampling run, concurrent embedded-host ownership or
GPU execution. Ordinary dispatch does not activate this checked method.
