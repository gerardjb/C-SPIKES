# Checked GCaMP CPU API contract

This opt-in validation surface is enabled by `C_SPIKES_TEST_CHECKED_CPU=ON` with `PGAS_BUILD_GPU=OFF`. Normal builds and existing constructors/methods retain their behavior. The option also enables test-only fault hooks; it is not a production activation switch.

## Ownership and results

Callers explicitly declare `checked::Context::owned_isolated_process()`. They must own exclusive use of the same GSL instance for the entire operation, including all threads and libraries. This declaration does not detect other users. The scoped handler captures bounded diagnostics without throwing or allocating and restores the exact previous handler on return or ordinary C++ unwinding. Missing context and nested checked calls return `context_required` and `context_busy` before replacing the handler. No local mutex can establish arbitrary embedded-host safety.

`try_create_gcamp(array<double,14>, CellParameters, Context)` returns a report and unique model or null. Five instance methods are `try_set_params(CellParameters, Context)`, `try_set_state(double, Context)`, `try_init(Context)`, `try_set_params_and_init(CellParameters, Context)`, and `try_steady_state(double, Context)`. Mutators return an UpdateResult; equilibrium returns a report and optional 12-state array. Status enums retain input, backend, factor-quality, rank/nullity, normalization and invariant distinctions. Reports include solve diagnostics and bounded callback context. Callback/return-code disagreement is failure.

Python exposes `(report, model_or_None)`, mutator reports and `(report, owned_array_or_None)` respectively. Cell-parameter methods take six scalar arguments in order G_tot, gamma, DCaT, Rf, gam_in, gam_out, followed by context. The factory takes float64 1D Gparams and Cparams arrays, then context. Existing array conversion and type/shape rules apply.

## Transaction and lifetime

The object address/lifetime remains stable. After successful checked updates, reacquire all borrowed native storage references; no successful-update storage-address guarantee is offered. Reported numerical failure leaves complete original contents and borrowed storage references unchanged. There is no simultaneous-reader/update guarantee.

Parameter updates commit six parameters, three fluorescence normalization values and final rate/matrix caches together; they retain state, Cparams, DFF and histories. State updates commit state, initial_state and caches. Initialization additionally zeros DFF. The composite stages parameter update plus initialization and commits once. Equilibrium preserves the existing cache-update behavior on success and publishes no state on failure. Histories retain values/shapes and are not recomputed.

Only the new factory initializes Cparams to six raw supplied values. Existing setters still do not synchronize Cparams; legacy constructors are unchanged. The checked factory uses a deferred private construction path and never first runs the old equilibrium solver. The 12-state order is nine GCaMP concentrations followed by BCa, Ca and Ca_in; concentration units are M.

This is numerical failure atomicity, not universal exception atomicity: existing type/resource/I/O exceptions are not redefined. Python result allocation can fail after a successful native commit. Status checks do not constitute a comprehensive scientific parameter-domain validator.

## Validation limits

Focused CPU contract fixtures cover successful commits, injected-failure rollback,
native lifetime/history preservation, Python ownership and aliases, handler
restoration/callback capture/unwinding, and absent/nested ownership. Injected faults
test propagation rather than actual SVD nonconvergence. Test hooks and observers
are excluded from normal builds. See [the fixture notes](../tests/native/contracts/README.md)
for setup and limitations.

These contracts do not establish GPU execution, inference quality, concurrent-host
safety, arbitrary embedded-Python safety, allocation-failure behavior or performance.
The current implementation still requires Armadillo; checked execution remains opt-in.

## Checked factory evolution default

Successful `try_create_gcamp` returns a model configured as `FIXEDLA` (linear-approximation endogenous buffer with the existing explicit stepping profile). The Python factory inherits this setting. Indicator equilibrium is independent of this evolution configuration. The factory publishes the same 12-state initialization and normalization values as before. Existing constructors retain their current default; checked instance methods preserve the caller-selected mode. Existing constructors and checked factories therefore have distinct evolution defaults; select the desired mode explicitly when comparing them.
