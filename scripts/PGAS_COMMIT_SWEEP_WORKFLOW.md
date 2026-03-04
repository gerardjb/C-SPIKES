# PGAS Commit Sweep Workflow (Slurm)

This describes how to run commit-by-commit PGAS comparisons using:
- `scripts/pgas_sbatch_template.sbatch`
- `scripts/pgas_commit_builds.json`
- `scripts/submit_pgas_commit_builds.py`

The goal is to run inference for each target commit into distinct run tags so results can be compared in GUI visualizations.

## 1. What each file does

1. `pgas_sbatch_template.sbatch`
   - Slurm job template that accepts two positional arguments:
     - `$1`: run tag
     - `$2`: commit hash
   - For each job, it:
     - creates a conda env named from run tag
     - checks out the requested commit into a detached worktree
     - performs a non-editable install (`pip install <worktree>`)
     - runs PGAS inference into run-scoped output/cache directories

2. `pgas_commit_builds.json`
   - Commit matrix with short run tags + full commit hashes.
   - You can add/remove/relabel entries safely.

3. `submit_pgas_commit_builds.py`
   - Reads the JSON matrix and submits one `sbatch` command per entry.
   - Supports filtering, dry-run, prefixes/suffixes, and error handling.

## 2. One-time setup before submission

Edit `scripts/pgas_sbatch_template.sbatch` and replace placeholder paths with your local system values:

- `REPO_ROOT` (or can be blank for root)
- `DATA_ROOT`
- `RESULTS_PARENT`
- `SLURM_LOG_DIR`
- `EDGES_PATH` (or leave placeholder to skip)
- `TRIAL_SELECTION_PATH` (or leave placeholder to skip)
- `DATASET_TAGS` (array of dataset stems)

Also check:
- `PGAS_CONSTANTS`, `PGAS_GPARAM`
- `BM_SIGMA`, `BM_SIGMA_GAP_S`
- `PYTHON_VERSION`

Important:
- The template removes an existing conda env with the same name and clears an existing build dir for that run tag.
- Use unique run-tag prefixes for different sweeps.

## 3. Ensure commits are available in the submitting clone

Before running any scripts, fetch the branch containing the commit hashes to ensure they're available to the slurm scheduler:

```bash
git fetch origin calibrate_bm_s2
```

If hashes in the json end up resolving in another branch, fetch that branch too.

## 4. Dry-run to test the process

```bash
python scripts/submit_pgas_commit_builds.py --dry-run
```

If template placeholders are still present and you intentionally want to test command rendering:

```bash
python scripts/submit_pgas_commit_builds.py --dry-run --allow-unresolved-template
```

## 5. Submit full sweep

```bash
python scripts/submit_pgas_commit_builds.py --run-tag-prefix <informative prefix>
```

Useful variants:

```bash
# Submit only selected run tags
python scripts/submit_pgas_commit_builds.py --include-run-tag b07_auto_calib --include-run-tag b12_provenance --run-tag-prefix <subset_run_prefix_tag>

# Exclude selected run tags
python scripts/submit_pgas_commit_builds.py --exclude-run-tag m00_main_base --run-tag-prefix <subset_excludes_>

# Add cluster-specific sbatch options
python scripts/submit_pgas_commit_builds.py --run-tag-prefix cmp_ --sbatch-arg=--account=<account> --sbatch-arg=--partition=<partition>

# Throttle submissions
python scripts/submit_pgas_commit_builds.py --run-tag-prefix <informative_prefix> --sleep-seconds 1.0
```

## 6. Output layout and GUI comparison

Each run writes under:

`<RESULTS_PARENT>/<run_tag>/`

with:
- `inference_cache/`
- `pgas_output/`
- `cli_evaluation/`
- `slurm/`

To compare in GUI:

1. Launch GUI from main build (though in theory all of the gui renderings from the worktree commits should not have cache collision issues).
2. Set dataset directory to the same `DATA_ROOT` parent used in the jobs.
3. In **BiophysSMC viz**, refresh run tags and select sweep run tags (for example `cmp_*`).
4. Compare trajectories/parameter traces across runs.

## 7. Recommended reproducibility approach

1. Use a consistent run-tag prefix per sweep batch.
2. Keep constants/gparam/edges/trial-selection fixed across commits.
3. Archive Slurm stdout/stderr with the run.
4. Avoid modifying the template mid-sweep (unless you're an asshole).

