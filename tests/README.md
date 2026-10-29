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
Review their workload and resource requirements before execution.

`profiles.json` is the authoritative file inventory; `TEST_DISPOSITIONS.json`
explains retained, replaced and consolidated coverage. Temporary port-parity
wrappers and their environment opt-in have retired; their useful cache/noise/window
assertions now live in `test_contract_*`. The earlier loose real-data PGAS test
has been replaced, not preserved as an independent numerical oracle.

`native/contracts` provides explicit CPU probes and an integrity-checked runner
for its listed subset. `native/history_contracts` provides independent history,
restart and writer fixture/scorer kernels. Neither directory is default pytest.
See their READMEs and `native/REFERENCE_REQUIREMENTS.md` for build, reference and
controller prerequisites. Historical identifiers are stable fixture labels, not
additional supported APIs. Do not regenerate expectations from the binary under
test or convert missing references into skips.

The native qualification controller, reference adapters and profile routing must
be validated on the completed source tree. Earlier execution of the assertions
does not validate a newly assembled runner, wheel or dependency combination.
Runtime/provider collision checks, installation/relocation, GPU smoke and matched
inference performance require their own explicit profiles and resource plans.
They are not implied by a passing default pytest invocation.
