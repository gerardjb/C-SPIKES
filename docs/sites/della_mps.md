# Della: verified site-managed MPS setup

Della documents `#SBATCH --gpu-mps`. For the tested resource class, use one `intel&gpu40` allocation with one GPU slice, two physical CPU cores and 8 GiB host memory. Della automatically routes the submission; do not hard-code `--partition=gpu` for this example. Verify current site configuration before submission, and set the time limit from measured workload duration plus cleanup margin. [Princeton GPU guidance](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing), [Della resources](https://researchcomputing.princeton.edu/systems-and-services/available-systems/della).

Prepare the [general launcher manifest](../pgas_launcher.md) first, with `--workers 2 --mps require --cpu-placement separate`. Load your existing CUDA/runtime environment and use the same Python installation and native binary that were pinned during preparation. The [batch example](../../examples/slurm/della_mps.sbatch) contains no personal account or filesystem paths:

```sh
sbatch --account=YOUR_ACCOUNT --time=00:30:00 examples/slurm/della_mps.sbatch \
  /absolute/frozen.json /absolute/results/run1
```

The example's 30 minutes is a template, not an estimate for an arbitrary queue. It requests one slice once, not one slice per worker. For recovery, append `--resume --retry-failed` to the same command with a new allocation. No other GPU class is substituted by the script.

The batch-level flag starts the site service before the script. `python -m c_spikes.slurm_mps --output NEW_DIRECTORY` discovers only same-user daemons in the current job cgroup and verifies the pipe endpoint's PID, process start identity, service CPU scope and assigned GPU. Explicit daemon GPU visibility is accepted when it matches; otherwise the helper requires enabled device cgroups and a no-context, unfiltered enumeration in the **identical service cgroup** that exposes exactly the assigned UUID. This matters because login-node configuration may differ from the compute node. The helper writes diagnostics before returning failure. It never starts, quits or reconfigures a service.

Pass the resulting `service-scope.json` to `--mps-service-file`. Every real inference process rechecks the scope before native import, then verifies MPS-enabled status and actual server membership. A daemon or an accepted flag alone is insufficient. The first active group has a joint startup barrier; each replacement is independently gated. [NVIDIA control interface](https://docs.nvidia.com/deploy/mps/mpsv2-interface.html).

The tested site service used the default pipe directory, with scope established from the actual job-owned endpoint and effective device restrictions. Do not replace it with a private pipe variable in the script and assume that the already-started daemon inherited that variable. Do not contact an endpoint from another job, change parent GPU compute modes, or reconfigure MIG. This route has not established coexistence of multiple MPS jobs from the same Unix user on one node. Refuse an ambiguous scope and consult Research Computing rather than launching an alternate daemon.

The launcher retains diagnostics under the run's output root, terminates only its own worker groups on cancellation, and leaves service shutdown to Slurm. Scientific settings and outputs belong to the general manifest; Della-specific configuration is confined to this setup and batch example. Campaign reports and benchmark data are maintained separately from the launcher PR.
