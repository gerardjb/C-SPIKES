# Native-core installation and relocation checks

Explicit Linux CPU qualification, not default pytest. MATRIX.json defines 80
fresh processes: 64 PGAS host/import-order/collision checks, eight OASIS golden
comparisons and eight package-conformance checks, split evenly between prefixes
A and B. The first prefix is moved and must be absent during relocated tests.

Prepare a new disposable runtime root; never run qualify.py in this source
folder. Copy the files mapped by INPUTS.json to their runtime-relative locations,
verifying each source hash first. The two OASIS test modules reuse existing
repository goldens rather than duplicating them. Do not use Python optimization,
which disables fixture assertions. No scheduler or dependency installer is supplied.

Configure separate ordinary and checked CPU builds from the current source,
with the explicit provider profile and native install options documented in
`docs/native-core-installation.md`. Include probes.cmake through
CMAKE_PROJECT_c_spikes_INCLUDE; it requires CMake 3.19+ and adds the g4a_header and
g4a_signatures compile probes plus linker maps. These names are stable test labels.
Build the collision.c DSO as collision.so in the runtime root. Install into
A/default and A/checked; if OASIS is built once, explicitly copy its newly built
installed bytes to the checked prefix. Build directories are build/default and
build/checked, with compile_commands.json, link.txt and maps retained.

Before execution, independently admit interpreter, modules, dependencies and
external runtimes by hashes. Verify the notice manifest for the actual dependency
versions. Use one-thread OMP/OpenBLAS/MKL/NumExpr settings, CPU PGAS with host
resampling/full trajectory, no preload or source-tree PYTHONPATH, and no provider
build directory in loader search paths. Both installed prefixes must be new test
artifacts, B and results must not exist, and tests must not mutate a user install.

The build_proofs.py kernel inspects explicit exports, local numerical symbols,
XERBLA/CBLAS ownership, dependency/object manifests and loader/RPATH state. Its
14-translation-unit and 30-byte handler assertions are source/provider-specific
recorded expectations, not general requirements on every future toolchain.
Reconcile them against the actual admitted source/provider before use; do not
silently weaken or retain a stale count. It requires readelf and ldd plus proofs/.
Do not run ldd on untrusted binaries.

After build proofs, write share/c_spikes/ARTIFACT_MANIFEST.json in each prefix
with its variant, an observations Boolean matching the compiled mode, and a files mapping of every installed relative path (except
that manifest itself) to SHA-256. Admit those manifests, the scripts, fixtures,
collision.so and external-runtime hashes before starting the controller.
The qualify.py kernel then runs fresh children and moves A to B. It requires
G2AD_DEADLINE as a monotonic-clock deadline computed by the same host controller,
30-second startup and 10-second body limits plus cleanup reserve. An outer bounded
controller must handle interruption/cleanup and preserve every receipt. Do not
reuse an old deadline or run without that controller. Failure stops the pass.

The assertions and matrix are retained from recorded checks; this relocated
source packaging and any new controller are not execution-validated yet. Final
validation must admit the final code and diagnostics mode, including exact export
inventory changes, and verify binary/input hashes before and after relocation.
No source/build fallback, oracle regeneration, missing-reference skips or automatic
runtime installation is permitted. A historical pass does not certify a new wheel
or arbitrary host. Optional-package and GPU checks remain separate.
