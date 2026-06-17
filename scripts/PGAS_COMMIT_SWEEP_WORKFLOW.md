# PGAS Allan Commit Sweep Workflow

This sweep plumbing targets the Allan b02 ladder only. The build matrix uses real commits from this branch, not overlay builds from the earlier `calibrate_bm_s2` worktree.

## Build Matrix

The matrix is `scripts/pgas_commit_builds.json`.

| run_tag | commit | conceptual coverage |
| --- | --- | --- |
| `m00_main_base` | `b6e7d799f564d567e1273b61ad624c473c5a8fca` | `m00_main_base` |
| `a00_b02_base` | `5aba209bbb12f44aee0dc1381a59ab8f27a309c7` | `a00_b02_base` |
| `a03_allan_bounds_cli` | `317141e60e2849fd2514f7c8a74aab59c0683c76` | `a03_allan_bounds_cli` |
| `a04_allan_calib_provenance` | `59d148a4e091f489010ebc3da08b8ded69abc41e` | `a01_b02_calib_scaffold`, `a02_b02_allan_calib`, `a04_allan_s2_provenance` |
| `a05_a06_allan_mh_logrw_widths` | `fa25284dcbe49b8e488f70fea758716c69cf61cf` | `a05_allan_mh_logrw`, `a06_allan_mh_widths` |

The `features` field controls which CLI flags the sbatch template passes. This keeps older commits from failing due to newer driver flags.

## Dry Run

```bash
python scripts/submit_pgas_commit_builds.py --dry-run
```

Submit one build:

```bash
python scripts/submit_pgas_commit_builds.py \
  --include-run-tag a05_a06_allan_mh_logrw_widths \
  --run-tag-prefix test_
```

Submit the full Allan ladder for a specific dataset/trial selection:

```bash
python scripts/submit_pgas_commit_builds.py \
  --run-tag-prefix s348_allan_ \
  --sbatch-arg=--export=ALL,DATASET_TAGS=jGCaMP8f_ANM478348_cell01,TRIAL_SELECTION_PATH=/path/to/trial_selection.json,BM_SIGMA=auto,BM_SIGMA_MAX=0.5
```

Use `--sbatch-arg=...` with an equals sign when the forwarded Slurm argument starts with `--`; otherwise `argparse` will treat the forwarded value as a submit-script option.

If you need a null edge path, leave `EDGES_PATH` unset or set it to an empty string in `--export`. The template only passes `--edges-path` when `EDGES_PATH` is non-empty.

## Important Environment Overrides

- `REPO_ROOT`: source worktree used for `git worktree add`; defaults to this repository.
- `DATA_ROOT`: Janelia-style input data root; defaults to `sample_data/janelia_8f/excitatory`.
- `RESULTS_PARENT`: parent for run outputs; defaults to `$DATA_ROOT/spike_inference`.
- `DATASET_TAGS`: whitespace-separated dataset stems. If omitted, dataset keys are inferred from `TRIAL_SELECTION_PATH`.
- `TRIAL_SELECTION_PATH`: JSON mapping dataset stems to trial indices.
- `EDGES_PATH`: optional edges `.npy`. Empty means no edge file is passed.
- `FIRST_TRIAL_ONLY`: set to `1` to run trial index `0` only. For commit-sweep compatibility, the template generates `$RUN_ROOT/slurm/first_trial_selection.json` and passes it as `--trial-selection-path` when no explicit `TRIAL_SELECTION_PATH` is supplied.
- `BM_SIGMA`: fixed value or `auto`; default is `auto`.
- `BM_SIGMA_MIN` and `BM_SIGMA_MAX`: bounds passed only to commits that expose them.
- `PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK`: set to `1` to pass the low-activity-mask flag where supported.
- `PGAS_SIGMA2_TARGET`, `PGAS_SIGMA2_ALPHA`, `PGAS_SIGMA2_PRIOR_STRENGTH`: sigma2 prior knobs passed only where supported.
- `CONDA_BASE_ENV`: conda env cloned for each build; default is `c_spikes_e`.
- `KOKKOS_SOURCE_OVERRIDE`: local `kokkos-src` path if auto-discovery under `$REPO_ROOT/build` is insufficient.
- `REUSE_EXISTING_BUILD`: set to `1` by default. When enabled, the template searches `$RESULTS_PARENT/_builds/*_<commit_short>` for a detached worktree at the requested commit plus a matching `<run_tag>_build` conda env, and reuses it instead of rebuilding.
- `FORCE_REBUILD`: set to `1` to ignore reusable builds and recreate the requested run-tag build/env.

## Cache Behavior

Each run writes to `$RESULTS_PARENT/$RUN_TAG` with run-scoped `inference_cache`, `pgas_output`, and `cli_evaluation` directories.

The template defaults `REFRESH_CONSTANTS_CACHE=1`, which removes only `$RUN_ROOT/inference_cache/pgas_constants` before running. This avoids stale derived constants when rerunning a tag with changed proposal-width or calibration defaults.

## Build Assumptions

The template expects `vcpkg` to exist at `$REPO_ROOT/vcpkg`, usually as a symlink to a shared install. It links that path into each detached build worktree so the pyproject-relative toolchain path resolves.

Slurm stdout/stderr defaults to `pgas_allan_commit_<jobid>.out/.err` in the submission directory. Override with `--sbatch-arg=--output=...` and `--sbatch-arg=--error=...` if you want centralized logs.

The sbatch template resolves `REPO_ROOT` from explicit `REPO_ROOT` first, then from `SLURM_SUBMIT_DIR` when submitted from a git worktree, then from the script path. This avoids Slurm-spooled script paths such as `/var/spool/slurmd` becoming the inferred repository root.

The template reuses existing builds by default. This lets a sweep job for `a04_allan_calib_provenance` reuse an already-built directory such as `$RESULTS_PARENT/_builds/s348_allan_a04_allan_calib_provenance_59d148a4` and its `s348_allan_a04_allan_calib_provenance_build` conda env. To force a clean rebuild, set `FORCE_REBUILD=1` through `--export` or use the sweep wrapper's `--force-rebuild`.

You can test first-trial selection generation without running a build:

```bash
RESULTS_PARENT=/tmp/pgas_first_trial_test \
DATASET_TAGS="cellA cellB" \
FIRST_TRIAL_ONLY=1 \
PGAS_TEMPLATE_TEST_FIRST_TRIAL=1 \
bash scripts/pgas_sbatch_template.sbatch test_first 5aba209 auto_bm
```

## Build Once, Run Many

For bm/sigma2 diagnostic sweeps, use `scripts/pgas_inference_sweep.json` plus the sweep submit wrapper. This builds the selected commit once, then runs every config listed in the sweep JSON inside the same job allocation.

Dry-run the current a04 diagnostic sweep:

```bash
python scripts/submit_pgas_inference_sweep.py --dry-run
```

Submit it:

```bash
python scripts/submit_pgas_inference_sweep.py
```

Force a new build instead of reusing a compatible build:

```bash
python scripts/submit_pgas_inference_sweep.py --force-rebuild
```

Resume a partial sweep by excluding completed config tags:

```bash
python scripts/submit_pgas_inference_sweep.py \
  --exclude-config-tag bm0p10_s2p0025_p5000
```

Run only selected configs:

```bash
python scripts/submit_pgas_inference_sweep.py \
  --include-config-tag bm0p10_s2p0035_p5000 \
  --include-config-tag bm0p15_s2p0035_p5000
```

The default sweep JSON targets `a04_allan_calib_provenance` and writes run roots like:

```text
sample_data/janelia_8f/excitatory/spike_inference/s348_diag_a04_bm0p10_s2p0025_p5000/
```

The current exploratory grid covers:

```text
BM_SIGMA_MAX: 0.10, 0.15
PGAS_SIGMA2_TARGET: 0.0025, 0.0035
PGAS_SIGMA2_PRIOR_STRENGTH: 5000, 10000
```

To run the same JSON against a different compatible build, override the build tag:

```bash
python scripts/submit_pgas_inference_sweep.py \
  --build-run-tag a05_a06_allan_mh_logrw_widths \
  --sweep-base-run-tag s348_diag_a05a06
```

To inspect commands without Slurm or building:

```bash
python scripts/run_pgas_inference_sweep.py \
  --sweep-json scripts/pgas_inference_sweep.json \
  --features auto_bm,bm_bounds,low_activity_mask,sigma2_prior \
  --dry-run
```
