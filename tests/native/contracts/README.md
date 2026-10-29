# Checked CPU contract fixtures

These opt-in fixtures protect checked-update failure atomicity, output ownership,
GSL handler restoration, context rejection, factory mode selection, diagnostic
context and checked sweep preparation. They require exclusive ownership of the
GSL instance in isolated processes. Injected failures test propagation and rollback;
they do not establish actual solver nonconvergence or universal exception safety.

Use a separate CPU-only build with the repository's admitted dependencies and
`C_SPIKES_TEST_CHECKED_CPU`, `C_SPIKES_TEST_SWEEP_CPU` and
`C_SPIKES_TEST_SWEEP_RUN_CPU` enabled. The optional `probes.cmake` hook requires
CMake 3.19 or newer and can be supplied through
`CMAKE_PROJECT_c_spikes_INCLUDE` when the configured project name is `c_spikes`.
It adds six `csp_contract_*` executables without changing the normal build.
This hook and the current fixture composition have not yet been execution-validated.

Run every case in a fresh process. `checked_transactions` takes a case ID
(T01–T11, T15, T16), then paths to `fixtures/gparams.dat` and `fixtures/cparams.dat`.
`diagnostic_context` takes D01–D06 with those same paths. `collector` takes only
the two parameter-file paths and checks its finite/invalid and unwind cases.
`factory` accepts P01–P04, C01 and C02 followed by those two paths.
`checked_python_case.py` takes the explicit checked extension path, T12–T14 and
the two fixture paths. `factory_python_case.py` takes C03/C04, the extension build
directory and the two fixture paths. These scripts are subprocess entrypoints,
not modules to import during default pytest collection.

`sweep_preparation` takes P01–P08 or T01–T08; `sweep_run` takes P01–P08 or F01–F08.
Both take the GCaMP parameter path and `fixtures/sweep.json` after the case ID.
The sweep fixture has two particles and three samples; it explicitly initializes
otherwise indeterminate fields. It is a bounded single-sweep correctness fixture,
not an inference or performance workload. Use one OpenMP thread, default host
resampling/full trajectory, and a temporary working directory; check that no
unexpected output files are created.

Successful exit verifies only in-process assertions. Paired numerical/RNG output
comparisons also require independently identified reference outputs and their
scorer; these must not be replaced by outputs from the executable under test.
The portable reference-scoring and collection configuration must be validated
before describing this directory as a complete runnable regression suite.

The existing CPU ancestor/selected-trajectory and Python options tests remain
separate. Owned-storage/provider, full inference and GPU qualification are not
implied by these fixtures. Preserve failure ordering, numerical envelopes and
fresh-process diagnostic IDs when adapting their storage interfaces.
