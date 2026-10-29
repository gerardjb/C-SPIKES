# Checked equilibrium mathematical unit

`checked_equilibrium_math.cpp` implements `checked::detail` using standard
fixed-size arrays and GSL/CBLAS, separately from model transactions. It does not
require model or Armadillo headers. The default native module also no longer links
libarmadillo; test-only historical reference headers remain a separate dependency.

The extraction preserves signatures, iteration order, status values, rank
thresholds, normalization rules and numerical envelopes. Scoped GSL error handling,
context ownership, model transactions and factories remain in
`checked_equilibrium.cpp`. The mathematical unit does not own the GSL handler;
callers must establish the documented exclusive context. Checked CPU execution
is opt-in and disabled by default.

The [native contract fixtures](../tests/native/contracts/README.md) include
mathematical and transaction assertions. Injected failures test propagation, not
actual backend nonconvergence. Standalone compilation or small numerical fixtures
do not establish GPU behavior, full inference accuracy or arbitrary embedded-host
safety. Test tolerances must retain their mathematical meaning, independently of
which storage or provider implementation is used.
