# Independent-fit PGAS launcher

`python -m c_spikes.pgas_pool` runs a bounded queue of **already independent, complete inference windows** on one assigned GPU. It starts a fresh interpreter for every fit and returns results in manifest order. Existing inference APIs remain serial by default; this launcher also defaults to one worker and MPS off. It does not split continuous recordings, change the model, calibrate across windows, select another device, or manage an MPS service.

This interface targets Linux, Python 3.9+, and a Slurm allocation with an explicit CPU count and exactly one visible GPU. MPS verification additionally requires a site-managed service whose job, device and CPU scope can be verified. See [Della setup](sites/della_mps.md) for the tested site route. General orchestration and manifests contain no Della paths, account names or resource classes.

The verified on/off modes require a driver exposing `CU_DEVICE_ATTRIBUTE_MPS_ENABLED`; an unavailable attribute is an error. Device-cgroup enumeration used by the site helper requires CUDA driver API 12.8 or newer. Driver/library versions and GPU class still need recording for performance comparisons; a frozen manifest does not make distinct hardware equivalent.

## Prepare before requesting a GPU

Make a JSON spec containing existing window files. Paths are relative to the spec file unless absolute. Each NPZ contains finite one-dimensional `time` and `fluorescence` arrays of equal length, with strictly increasing time. Supply the sampling rate used by the existing inference workflow; the launcher does not silently replace it with an estimate.

```json
{
  "fits": [
    {
      "fit_id": "cell01_epoch00",
      "replicate": 0,
      "input_file": "windows/cell01_epoch00.npz",
      "constants_file": "constants.json",
      "gparam_file": "sensor.dat",
      "raw_fs": 120.0,
      "config": {"niter": 200, "burnin": 100, "keep_output_dat_files": true}
    },
    {
      "fit_id": "cell02_epoch00",
      "input_file": "windows/cell02_epoch00.npz",
      "constants_file": "constants.json",
      "gparam_file": "sensor.dat",
      "raw_fs": 120.0,
      "config": {"niter": 200, "burnin": 100, "keep_output_dat_files": true}
    }
  ]
}
```

Add as many independent fits as needed. Choose the scientific settings and native binary appropriate for your existing workflow; the numbers above are examples, not new defaults. Prepare a frozen manifest on a CPU node:

```sh
python -m c_spikes.pgas_pool --prepare fits.json \
  --binary /absolute/path/pgas_bound_gpu.cpython-310-x86_64-linux-gnu.so \
  --manifest frozen.json --workers 2 --mps require --cpu-placement separate
```

Preparation loads CPU dependencies, never the native GPU extension. It expands the installed `PgasConfig` defaults, records the seeded base constants and Python/NumPy/SciPy versions, derives each CPU seed, and hashes the inputs, constants, kinetic parameters, native binary and Python implementation. The existing pipeline applies configuration overrides and records derived noise settings in fit metadata. Preparation refuses to overwrite a frozen manifest. Configuration fields controlling output paths and caches belong to the launcher. `edges` and ground-truth-dependent low-activity noise masking are unsupported here: supply windows that already have the intended independence and boundaries, or use the existing inference API.

The CPU seed is `stable_seed(fit_id, replicate)` and does not depend on queue position, worker number or completion order. The unchanged native GPU generator uses seed 42/reset per sweep; different CPU replicates do **not** imply independent GPU random streams. These controls are recorded in manifests and receipts. Automatic noise calibration remains the existing pipeline's operation on each supplied fit, with its effective results recorded in the fit metadata.

## Run the queue

Inside the allocation, using the same pinned installation:

```sh
python -m c_spikes.pgas_pool --manifest /absolute/frozen.json \
  --output /absolute/results/run1 \
  --mps-service-file /absolute/current-job/service-scope.json
```

Omit the service file for a manifest prepared with `--mps off`. Run-time `--workers`, `--mps` and `--cpu-placement` may restate frozen values; incompatible values are rejected before inference. The Python API is `run_manifest(manifest, output, ..., service_file=...)` in `c_spikes.pgas_pool`. The coordinator uses only the standard library and does not initialize CUDA.

- `shared`: all workers and the coordinator use the first allocated physical core.
- `separate`: workers occupy distinct physical cores. As a fit finishes, the next queued fit inherits the freed slot's core. Fast workers can continue while a slow fit is still running.
- `--worker-cpus 7,27` selects logical CPU IDs for the slots; these must be allocated, and separate placement rejects SMT siblings. `--coordinator-cpu` selects an allocated coordinator core; otherwise it uses the first slot. Numerical-library thread limits are one per worker.

The actual allocated core numbers, GPU UUID, job, affinity and service details are attempt provenance. They are not stable task identity, so resume can occur in a new allocation with different core numbers. The **placement policy, worker count, MPS mode, default CUDA wait policy, scientific configuration, seeds and file hashes** are stable identity. Use a new output root for an intentional change of those settings. Changing the installation requires preparing a new manifest.

`require` validates the service scope before importing the native extension. Every actual worker must report the assigned device, default scheduling flags, MPS-enabled driver attribute and membership in the intended job-owned server. The initial active group waits for simultaneous verification before inference. Replacements wait for their own verified gate; they do not wait for long-running peers. Attachment is checked again at completion. A failed gate stops admission and cancels owned workers; a normal inference failure is isolated and the remaining queue continues. There are no automatic retries.

`off` uses a new empty private communication directory and verifies MPS-disabled status; it never quits an existing daemon. Neither policy changes Slurm GPU visibility, CUDA scheduling flags, MIG geometry or parent GPU modes. The site scheduler remains responsible for the service lifecycle.

## Recovery and diagnostics

Each fit writes to `<output>/<fit_id>/attempt-NNNN/`, with its own input task, log, raw/cache paths and derived outputs. A child publishes `completion.json` atomically after inference, output validation and hashing. The coordinator then records successful process exit. **Both records are required for reuse**: a receipt followed by nonzero exit, a cancelled worker, or a receipt without a final coordinator record is incomplete.

```sh
python -m c_spikes.pgas_pool --manifest /absolute/frozen.json \
  --output /absolute/results/run1 --resume --retry-failed \
  --mps-service-file /absolute/new-job/service-scope.json
```

`--resume` checks identities and artifact hashes before reusing completed fits. `--retry-failed` permits one fresh attempt for incomplete, failed or damaged outputs, preserving old attempts. Changed task identity is always rejected. Successful cached fits are omitted from gates and count as zero newly completed inference fits. New attempts have fresh barrier tokens; tokens, attempt paths and service PIDs never participate in compatibility checks.

SIGINT/SIGTERM stops admission, terminates only this invocation's worker process groups, and records cancelled/not-started fits in `batch.json`. The CLI exits 130 for cancellation and 1 for failures. A root lock excludes concurrent coordinators; workers also hold the lock across an abrupt coordinator death. SIGKILL or node loss cannot guarantee final diagnostic writes; the next run treats missing success records as incomplete, and Slurm handles allocation cleanup. Resume does not append to a partially sampled chain.

Inspect `worker.log`, `worker-error.json`, `execution-start.json`, `mps-start.json`, `mps-end.json`, gate `decision.json`, and `process.json`. Driver errors, membership queries and stderr are saved before rejection; readable service-log tails are captured on failure. `run-*/batch.json` retains each invocation, while root `batch.json` is the latest summary. A completely cached invocation does not contact an MPS service. Scope discovery failures retain `service-scope.json` and any available logs.

## Validation and performance limits

CPU tests use real subprocesses with simulated CUDA evidence to exercise scheduling, identity, barriers, cancellation and recovery. They do not establish real CUDA attachment or throughput. The MPS mechanism has separately been measured with real independent inference processes; queues longer than the active worker count need their own GPU acceptance measurement before claiming production performance. A listed MPS client is evidence of attachment, not a kernel-overlap trace.

Budget GPU time and memory for the complete fits and the number of simultaneous workers. Report newly completed fits divided by occupied allocation time, including startup and failures; cached fits and summed overlapping worker latencies are not throughput. Keep the ordinary serial route available and select concurrency explicitly.
