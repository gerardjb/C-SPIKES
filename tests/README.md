# Test profiles

Run `python -m pytest` for `native_core`, after building the CPU extensions for the
current source and interpreter. This selects seven files covering OASIS numerical
behavior, tiny seeded PGAS smoke, state-history ownership and synthetic-writer
contracts. It does not run GPU inference or establish inference accuracy.

Use `python -m pytest --csp-profile=python_contracts` for ten adapter, cache,
windowing, noise-calibration and orchestration files. These may require NumPy,
SciPy and plotting dependencies; the profile name does not mean dependency-free.
Use `--csp-profile=optional_package` for broader GUI, command-line, queue and
optional-backend tests with their documented dependencies. `--csp-profile=all`
explicitly selects all Python profiles. Unexecuted profiles are not qualified.
Selection occurs before module imports so optional stacks do not become default
native-core collection requirements. Unknown profiles and empty final selections
are errors. Slow tests remain excluded unless explicitly selected with `-m slow`.
To include both slow and non-slow tests, use
`python -m pytest --csp-profile=all -o addopts=`.
Review their workload and resource requirements before execution.

`profiles.json` is the authoritative file inventory. The `test_contract_*` files
protect cache payloads/consumers, noise calibration and windowing without requiring
native inference. The seeded PGAS smoke checks schema and ownership; it is not an
independent inference-accuracy oracle.

`native/contracts` provides explicit CPU probes and an integrity-checked runner
for its listed subset; it is not default pytest. See its README for build,
reference and controller prerequisites. `run_supplemental.py` dispatches the additional checked factory/sweep/Python
fixtures separately. `C_SPIKES_TEST_RESAMPLING=ON` connects ancestor sampling,
selected trajectory, terminal-weight copying and diagnostic-header fixtures to
CTest; CUDA fixtures require a CUDA build and GPU. Historical identifiers
are stable fixture labels, not additional supported APIs. Do not regenerate expectations from the binary under
test or convert missing references into skips.

Use separate ordinary and checked build directories and the explicit dependency
and manifest admission described in `native/contracts/README.md`. Source-tree
results do not validate a wheel or a different dependency combination.
Runtime/provider collision checks, installation/relocation, GPU smoke and matched
inference performance require their own explicit profiles and resource plans.
They are not implied by a passing default pytest invocation.
