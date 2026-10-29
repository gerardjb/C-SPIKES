# Native CPU contract fixtures

These opt-in checks protect owned storage, column-major layout, parsing, signed
indices, gathers/reductions, numerical failure ordering, output ownership, aliases,
checked transactions and diagnostic context. The small serialized references are
comparison inputs, not inference datasets. Identifiers in `cases.json` are stable
fixture labels. `TEST_SOURCES.json` identifies the source revisions and hashes.

## Build and dependency contract

Use separate ordinary and checked CPU build directories. The checked profile needs
`PGAS_BUILD_GPU=OFF` and all three `C_SPIKES_TEST_CHECKED_CPU`,
`C_SPIKES_TEST_SWEEP_CPU` and `C_SPIKES_TEST_SWEEP_RUN_CPU` options ON. With project
name `c_spikes`, supply `CMAKE_PROJECT_c_spikes_INCLUDE` pointing to `probes.cmake`.
The hook requires CMake 3.19 or newer for deferred setup. It adds explicit
`csp_contract_*` targets and a standalone compile-assertion target, without changing
the normal product build.

Supply `CSP_CONTRACT_ARMADILLO_INCLUDE`, `Python_EXECUTABLE` and
`CSP_CONTRACT_PYTHON_HOME` explicitly. The latter must match the admitted embedding
interpreter and development library. Armadillo headers are an independent test
reference dependency; the production module does not link libarmadillo. Use the
pinned numerical-provider profile, GSL/CBLAS, JsonCpp, CPU Kokkos and compatible
compiler/Python/NumPy dependencies. Dependency installation and backend substitution
are not performed by the test runner.

Observational fixtures require diagnostics enabled if the build offers a switch.
Essential protections also require their own diagnostics-disabled checks; success
of the observational fixtures does not establish that behavior.

## Runner and exact comparisons

`run_contracts.py` requires an independently reviewed build manifest and its SHA-256,
explicit binary paths/hashes, the matching `INPUTS.json` identity, a new output
directory and a bounded deadline. Use `--build-manifest`,
`--expected-build-manifest-sha256`, `--output` and `--deadline-seconds`; omitting
`--run` performs admission only. The runner does not build or update reference
values. Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and
NUMEXPR_NUM_THREADS to 1; do not use Python `-O`.

The manifest describes 37 fresh processes, preserving grouped subcase order and
exact JSON/text comparisons. Multiline JSON framing, diagnostic-ID isolation,
child cleanup and fail-on-missing-reference behavior are retained. Failures keep
stdout/stderr and a ledger. This orchestration composition and build hook have not
yet been execution-validated. Hash manifests must be deliberately reviewed and
refrozen when source changes; do not regenerate them automatically during a run.

## Additional checked factory and sweep fixtures

`factory` takes P01–P04, C01 or C02, followed by the GCaMP and cell parameter paths.
`checked_python_case.py` takes the explicit checked extension path, T12–T14 and
those parameter paths. `factory_python_case.py` takes C03/C04, the extension build
directory and the parameter paths. They are subprocess entrypoints, not default
pytest modules. Run each case in a fresh process.

`sweep_preparation` takes P01–P08 or T01–T08; `sweep_run` takes P01–P08 or F01–F08.
Supply `fixtures/gparams.dat` and `fixtures/sweep.json` after the ID. The fixture
uses two particles and three samples, initialized named fields, host resampling,
full trajectory and one thread. Run in a temporary working directory and reject
unexpected output files. These checks cover preparation rollback and tiny single
sweeps, not inference accuracy or performance. The preparation fixture separately
protects returned handoff snapshot independence.

These additional fixtures are not included in the 37-process runner. Paired
numerical/RNG comparisons require independent reference outputs and their scorer;
the executable under test must not generate its own replacement oracle. Reference
routing and the complete collection configuration need validation before this is
represented as a complete regression suite. Injected failures do not establish
actual SVD nonconvergence or universal exception safety.

Existing ancestor/selected-trajectory tests protect distinct sampling/copy behavior.
Optional-package tests remain separate from native checks. No GPU, arbitrary
embedded-host, wheel portability or full inference qualification follows from this
CPU subset. Required third-party notices remain with the derived code and fixtures.
