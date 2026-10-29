# Optional numerical observations

Native PGAS builds default to `C_SPIKES_OBSERVATIONAL_DIAGNOSTICS=OFF`.
For an observational build, change the existing CMake argument in pyproject.toml
to `-DC_SPIKES_OBSERVATIONAL_DIAGNOSTICS=ON`, or pass that option to a native CMake
source build. Preserve the remaining compiler/toolchain arguments. Use separate
build directories and rebuild all PGAS objects and bindings together for each mode.
This is a compile-time choice, not a runtime TOML configuration file.

Both modes can be Release builds. The CPU or GPU extension's
`numerical_diagnostics_enabled` Boolean reports the compiled choice. Observations
include optional nonfinite scans, context capture and warning reports. OFF elides
observation arguments as well as scans; it does not remove input/index validation,
required exceptions, checked-solver validation, rollback, GSL handler restoration
or resource checks. The checked GSL API remains a separate opt-in. The choice
changes no model equation, RNG call or solver policy.

In an ON build, Python logging DEBUG and the native stderr switch
`C_SPIKES_NUMERICAL_DEBUG=1` select extra report detail. They cannot turn an OFF
build into an ON build. Scientific runtime configuration remains JSON.
[Diagnostic logging](numerical_diagnostics.md) describes interpretation and limits.

Native clients compiling headers directly default to OFF. To opt in, define
`C_SPIKES_OBSERVATIONAL_DIAGNOSTICS=1` consistently across every relevant
translation unit. Do not mix headers/objects built in different modes; the CMake
interface target propagates the setting to the project's PGAS targets.

`tests/native/test_observational_diagnostics.cpp` checks both compiled settings,
including argument non-evaluation in OFF, finite silence, ON aggregation/context
and exception preservation. It is a standalone header fixture, not default pytest.
Existing observation/context fixtures require ON. Required input, ownership,
error and rollback behavior must also be checked with OFF; passing an ON-only
suite does not cover it. Installation export checks include the Boolean and must
compare it with the admitted build-mode manifest.

A recorded same-window comparison found identical scientific outputs for the
examined ON/OFF runs. One comparison does not establish a general speedup or
broad toolchain/platform compatibility. Preserve separate configuration identities
and affected validation when rebuilding or distributing either mode.
