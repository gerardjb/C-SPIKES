# Extended native qualification inputs

The checked CPU runner in `contracts/run_contracts.py` uses the small references
already shipped beside it. Its explicit build manifest admits each binary and the
source/input hashes; it never builds or repairs an oracle. See its README for the
38-process subset and the separate factory/sweep fixtures.

Extended Python conversion/context, history formula, restart, writer, provider
and import-order checks also need an independent reference/controller package.
`EXTERNAL_REFERENCES.json` identifies the recorded controller inputs by SHA-256,
source fixtures by immutable Git revision and path, and reference directories by
canonical inventory digest. Obtain that package from the project's maintainers;
it is not downloaded or assumed present. To verify a reference directory, map each
regular file's package-relative POSIX path to its SHA-256, serialize the mapping
with sorted keys and compact JSON separators (`,` and `:`), then hash those UTF-8
bytes. Compare the file count and total bytes as well. Original Git paths can be
retrieved with `git show <source_revision>:<source_path>` when that history is
available. Access to historical investigation objects is not assumed for users.

These hashes identify source evidence, not an execution command for today's tree.
Historical controllers may contain machine paths and runtime layouts. Do not run
their scheduler/job wrappers or use their binary paths directly. Prepare a separate
runtime manifest with the fields listed in EXTERNAL_REFERENCES.json; map the
current default and checked binaries explicitly, verify hashes and module origins,
and review the controller adaptation before executing it. Retain fresh-process
isolation for diagnostic-ID-sensitive cases, complete RESULT JSON framing, startup
READY/GO handshakes and bounded child cleanup. Missing inputs or a wrong variant
are admission errors, never skip/pass conditions.

For history checks, `history_contracts/README.md` defines the fixture layout and
observer placement. Apply the observer only to a disposable, hash-matched test
copy. Endpoint and first-fluorescence scorers keep their independent formulas and
fixed bounds; historical uninitialized values are not valid numerical oracles.
Factory/sweep checks retain independent numerical/RNG references. Keep optional
Cascade/ENS2 reader cases outside the native-core dependency set.

Before claiming extended qualification for a new source tree, validate the adapted
controller and complete case/reference routing, confirm source-specific exports,
verify reference and binary identities, and run the affected checks under a
bounded execution plan. The shipped kernels and external evidence manifest do
not by themselves establish a portable, self-contained end-to-end test suite.
Unavailable historical tooling limits a reproduction claim; it does not justify
weakening present regression assertions. Saved receipt counts are not current
collection counts. GPU/provider/install and full-inference claims require their
own explicitly admitted configuration and results.
