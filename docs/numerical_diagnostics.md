# Numerical diagnostic logging

In builds with `C_SPIKES_OBSERVATIONAL_DIAGNOSTICS=ON`, GCaMP model calls can report nonfinite values or invalid fluorescence normalization. The diagnostic identifies the operation, first detected field/index, configured mode and actual integrator where known. It reports numerical observations; it does not establish their cause or prove a stability limit.

Observations default to OFF. See [build-mode selection](observational-diagnostics.md) for the CMake/TOML option and compiled-mode indicator. Runtime logging settings cannot enable observations compiled out of an OFF build.

This feature logs only. FIXED, FIXEDLA, their defaults, outputs, existing exceptions and numerical calculations are unchanged. Existing invalid values can still be returned or saved. Absence of a warning does not establish accuracy or scientific validity.

An affected outer call emits at most two WARNING records: its first detected symptom and a summary of detections/suppressed duplicates. Both records are deferred until the outer call completes or unwinds. Consequently, a long Analyzer run can finish before its warning appears. An abrupt process termination may prevent reporting. `nominal_h` is the configured fine-step size, not a claim about every floating-point interval. Substep/index are the first *detected* location, not necessarily the origin of the problem. Zero-calcium computations with finite resulting rates are not warned about merely because an intermediate logarithm is nonfinite.

Python-bound calls use the `c_spikes.pgas` Python logger without configuring global logging. Optional bounded context is available by setting that logger to DEBUG in the calling application. Native-only execution uses stderr; set `C_SPIKES_NUMERICAL_DEBUG=1` to include bounded DEBUG context there. Ordinary valid calls add no diagnostic messages. Logging-handler failures do not replace the result or existing exception.

Coverage includes native GCaMP construction, direct methods, steppers, history outputs and getters. Nested model calls aggregate within Analyzer/CLI host scopes. OpenMP parallel-worker fallback reporting is suppressed to avoid per-particle messages; **parallel host workers, particle kernels, and synthetic/cache adapters are not instrumented**. Analyzer coverage therefore does not imply end-to-end inference diagnostics. No GPU qualification is claimed. Counts describe observations, not independent failures; source version and seed are not captured automatically.

If useful, copy this information into a manually filed issue:

- Package version, source commit/build and platform.
- Operation, selected mode, requested interval and diagnostic records.
- Small reproduction call and parameter values needed to reproduce it.
- Expected behavior and observed result or existing exception.
- Optional data that you have reviewed and chosen to share.

No issue, telemetry or data upload occurs automatically. Avoid including credentials or private data in an issue.
