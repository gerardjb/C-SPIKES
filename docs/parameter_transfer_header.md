# Standalone parameter-transfer header

`src/c_spikes/pgas/include/GCaMP_params.h` defines the global `GCaMP_params`
structure without including the native model, Armadillo or Kokkos. Field ordering,
types, defaults and snapshot ownership are preserved. `GCaMP_model.h` includes it
for existing model consumers. Clients of `particle_array.h` that also need
`GCaMP` must explicitly include `GCaMP_model.h`; incidental transitive inclusion
is not a supported interface.

Layout and handoff checks compare initialized named members and offsets, not
padding or indeterminate values. This header boundary does not by itself establish
cross-compiler/device ABI compatibility, inference accuracy or general
embedded-host GSL safety. Existing ownership, failure and numerical policies apply.
