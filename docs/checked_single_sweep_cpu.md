# Opt-in single CPU sweep

SMC::try_PGAS(const param&,const Trajectory&,Trajectory&,const checked::Context&) is exposed only with OFF-by-default C_SPIKES_TEST_SWEEP_RUN_CPU, requiring CPU checked sweep preparation. It returns checked::SweepRun with preparation diagnostics and sweep_returned. It is not automatically used by Analyzer, PGAS or parameter proposals and has no Python binding.

The new call checks counter overflow using the existing exception before model mutation, then invokes checked preparation. Missing/busy context or numerical preparation failure returns before counter/RNG/particle/output mutation. Model contents and borrowed references follow the checked preparation rollback contract. On success model storage references must be reacquired, and the existing shared sweep body runs once. The original read_params position after reconstruction remains unchanged. The old PGAS entry point calls this same body with legacy preparation and original counter/preparation order.

No whole-sweep rollback is promised after preparation commits. Existing resource/type exceptions and warning/instability behavior remain. sweep_returned means execution returned normally, not proof of finite/valid inference output. No implicit rejection, retry, timestep/integrator switch, invalid-particle filter, file publication or new numerical exception policy is added. Output storage is caller-owned. Valid existing SMC construction, correctly sized trajectories and exclusive ownership in an isolated GSL process are preconditions; the context is a declaration, not an ownership detector.

Focused fixtures exercise tiny CPU single sweeps with initialized input fields,
named-state snapshots, RNG bytes and counters. Selected-state observation must
fail if ancestry reconstruction is ambiguous. Evolution-dependent matrix and log
weights are finite diagnostics, not quantities covered by a generic invented
parity tolerance. These checks do not establish full inference, statistical,
device or arbitrary embedded-host behavior. See the
[fixture notes](../tests/native/contracts/README.md).
