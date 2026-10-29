# History, restart and synthetic-output contract fixtures

These opt-in fixture and scoring kernels protect behavior that shape-only tests
cannot establish. They are not a ready-made default pytest suite. No historical
job wrappers, scheduler logs or duplicate production model source are included.
The final build/runtime adapter and case collection still require validation.

* `endpoint/`: twelve explicit binary64 time/spike configurations, preserving the
  existing absolute-time grid prefix and independently interpolating newly covered
  output. Its scorer retains the 32-epsilon local-scale rounding bound and exact
  unaffected values. The saved reference may contain historical endpoint NaNs;
  they are not expected output for the corrected implementation.
* `initial_fluorescence/`: four paired fixtures plus public-path and native
  controls. Sample zero is derived independently from the actual initial indicator
  state and captured calibration. Concentrations/grid/spike assignments and later
  defined fluorescence retain exact comparisons. The observer never reads the old
  uninitialized fluorescence sample zero.
* `restart_occupancy_writer/`: A01–A14 validate fresh/append saved-record counts,
  11-column restoration, status and no output mutation on invalid restart; O01–O08
  validate independent pool denominators, owned fractions and raw nonfinite data;
  W01–W12 validate the writer's schema, optional inputs and sanitization. The real
  Cascade/ENS2 reader cases W07/W08 require their optional package dependencies and
  must not become default native-core requirements.

`observer.h` and `OBSERVER.patch` describe the isolated native observation build.
`OBSERVER_INPUTS.json` pins the exact production and observed-source hashes. Apply
only to a disposable test copy after verifying its source hash; never patch the
production model in place. Both capture calls occur after evolution and before
interpolation. Regeneration on a different source requires placement review.

Python subprocess helpers accept an explicit prepared runtime root, case ID and
READY pipe descriptor (plus variant where required); they must be driven by a
bounded controller with startup/body/cleanup limits. The endpoint/fluorescence
runtime layout uses PLAN.json from fixtures.json, gparams.dat/cparams.dat,
source/src, explicit reference-binary metadata and build/candidate extensions.
The restart/occupancy/writer entrypoint uses cases.json, RUNTIME_PINS.json and
runtime_<variant>/src. Do not execute these helpers by importing them in pytest.
No runtime binary is supplied or admitted by this directory.

References are retained independently from the executable under test. Compare
parsed arrays/files/RNG as specified; do not regenerate expected values from the
candidate, normalize diagnostic IDs, restore obsolete zero/NaN expectations, or
replace missing references with skips. Reader roundtrip checks are not training
qualification. Tiny CPU sweeps are smoke/contract checks, not GPU inference or
performance evidence. Existing full-inference evidence has its own configuration.
