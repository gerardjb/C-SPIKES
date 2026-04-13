#!/bin/bash
#SBATCH --job-name=pgas_commit_eval
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES/logs/slurm_logs_build_test/output/%x_%j.out
#SBATCH --error=/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES/logs/slurm_logs_build_test/error/%x_%j.err

set -eo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <run_tag> <commit_hash>" >&2
  exit 2
fi

RUN_TAG="$1"
COMMIT_HASH="$2"
COMMIT_SHORT="$(echo "$COMMIT_HASH" | cut -c1-8)"
ENV_NAME="${RUN_TAG}_build"

# -----------------------------
# User-specific configuration
# -----------------------------
REPO_ROOT="/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES"
DATA_ROOT="/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES/sample_data/janelia_8f/excitatory"
RESULTS_PARENT="/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES/sample_data/janelia_8f/excitatory/spike_inference"
SLURM_LOG_DIR="/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES/logs/slurm_logs_build_test"
PGAS_CONSTANTS="$REPO_ROOT/parameter_files/constants_GCaMP8_soma.json"
PGAS_GPARAM="$REPO_ROOT/src/c_spikes/pgas/20230525_gold.dat"
EDGES_PATH="/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES/sample_data/janelia_8f/excitatory/edges/edges.npy"
TRIAL_SELECTION_PATH="/scratch/gpfs/WANG/gerardjb/temp/C-SPIKES/sample_data/janelia_8f/excitatory/spike_inference/run_3/slurm/trial_selection.json"
# If empty, datasets are inferred from TRIAL_SELECTION_PATH keys (when provided).
DATASET_TAGS=("jGCaMP8f_ANM471993_cell01" "jGCaMP8f_ANM471993_cell02")
USE_CACHE=1
BM_SIGMA="0.02"
BM_SIGMA_GAP_S="0.15"
CONDA_BASE_ENV="c_spikes_e"
CONDA_OFFLINE_MODE=1
KOKKOS_SOURCE_OVERRIDE="${KOKKOS_SOURCE_OVERRIDE:-}"

# -----------------------------
# Environment + build setup
# -----------------------------
source /etc/profile.d/modules.sh
module load anaconda3/2024.6
module load cudatoolkit/12.9
module load cmake/3.30.8

if [[ "$CONDA_OFFLINE_MODE" == "1" ]]; then
  export CONDA_OFFLINE=true
fi

mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$RESULTS_PARENT/_builds"

BUILD_DIR="$RESULTS_PARENT/_builds/${RUN_TAG}_${COMMIT_SHORT}"

echo "[setup] run_tag=$RUN_TAG commit=$COMMIT_HASH env=$ENV_NAME"

# Fresh env per run/build name for clean, non-editable installs.
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env remove -y -n "$ENV_NAME"
fi
if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_BASE_ENV"; then
  echo "[error] Base env '$CONDA_BASE_ENV' not found. Create it first on a login node." >&2
  exit 3
fi
conda create -y -n "$ENV_NAME" --clone "$CONDA_BASE_ENV"
conda activate "$ENV_NAME"

if ! command -v g++ >/dev/null 2>&1; then
  echo "[error] g++ not found. Load a compiler module before build." >&2
  exit 4
fi
if ! command -v gfortran >/dev/null 2>&1; then
  echo "[error] gfortran not found. Load a compiler module before build." >&2
  exit 4
fi

# Use a detached worktree so main branch state is untouched.
# Prune stale registrations immediately (default prune horizon can miss fresh stale entries).
git -C "$REPO_ROOT" worktree prune --expire now || true

# Clear any existing registration for this path, then prune again.
# This handles the "missing but already registered worktree" failure mode on reruns.
git -C "$REPO_ROOT" worktree remove --force "$BUILD_DIR" 2>/dev/null || true
git -C "$REPO_ROOT" worktree prune --expire now || true

if [[ -d "$BUILD_DIR" ]]; then
  rm -rf "$BUILD_DIR"
fi
git -C "$REPO_ROOT" worktree add --detach -f "$BUILD_DIR" "$COMMIT_HASH"
git -C "$BUILD_DIR" checkout --detach "$COMMIT_HASH"

# Worktree checkouts do not include untracked files from REPO_ROOT.
# Wire in vcpkg so pyproject's relative toolchain path resolves in this build.
if [[ ! -f "$REPO_ROOT/vcpkg/scripts/buildsystems/vcpkg.cmake" ]]; then
  echo "[error] vcpkg toolchain not found at $REPO_ROOT/vcpkg/scripts/buildsystems/vcpkg.cmake" >&2
  echo "[error] Install/clone vcpkg under REPO_ROOT before running commit sweeps." >&2
  exit 5
fi
ln -sfn "$REPO_ROOT/vcpkg" "$BUILD_DIR/vcpkg"
export VCPKG_ROOT="$BUILD_DIR/vcpkg"
export CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$BUILD_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake${CMAKE_ARGS:+ $CMAKE_ARGS}"

# Force offline Kokkos FetchContent to use a local source checkout.
if [[ -z "$KOKKOS_SOURCE_OVERRIDE" ]]; then
  if [[ -d "$REPO_ROOT/build/_deps/kokkos-src" ]]; then
    KOKKOS_SOURCE_OVERRIDE="$REPO_ROOT/build/_deps/kokkos-src"
  else
    KOKKOS_SOURCE_OVERRIDE="$(find "$REPO_ROOT/build" -maxdepth 5 -type d -name kokkos-src 2>/dev/null | head -n 1)"
  fi
fi

if [[ -n "$KOKKOS_SOURCE_OVERRIDE" && -d "$KOKKOS_SOURCE_OVERRIDE" ]]; then
  echo "[setup] Using offline Kokkos source: $KOKKOS_SOURCE_OVERRIDE"
  export KOKKOS_SOURCE_OVERRIDE
  export CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_KOKKOS=$KOKKOS_SOURCE_OVERRIDE -DFETCHCONTENT_SOURCE_DIR_Kokkos=$KOKKOS_SOURCE_OVERRIDE ${CMAKE_ARGS}"
else
  echo "[error] Local Kokkos source not found at: $KOKKOS_SOURCE_OVERRIDE" >&2
  echo "[error] FetchContent is offline on compute nodes; set KOKKOS_SOURCE_OVERRIDE to an existing kokkos-src path." >&2
  exit 6
fi

# Extra hardening for older/newer commits: rewrite Kokkos FetchContent source to local path.
DEPS_CMAKE="$BUILD_DIR/dependencies/CMakeLists.txt"
if [[ -f "$DEPS_CMAKE" ]] && grep -q "github.com/kokkos/kokkos.git" "$DEPS_CMAKE"; then
  echo "[setup] Patching dependencies/CMakeLists.txt to use local Kokkos source"
  sed -i "s|GIT_REPOSITORY https://github.com/kokkos/kokkos.git|SOURCE_DIR \"$KOKKOS_SOURCE_OVERRIDE\"|g" "$DEPS_CMAKE"
  sed -i '/^[[:space:]]*GIT_TAG[[:space:]]\+/d' "$DEPS_CMAKE"
fi

# Non-editable install from the commit snapshot.
python -m pip install --no-deps --no-build-isolation "$BUILD_DIR"

# Verify we can import the installed package and at least one PGAS backend.
unset PYTHONPATH
python - <<'PY'
import c_spikes
import c_spikes.pgas.pgas_bound as pgas_bound
print(f"[setup] c_spikes imported from: {c_spikes.__file__}")
print(f"[setup] PGAS backend selected: {getattr(pgas_bound, '__backend__', 'unknown')}")
PY

# -----------------------------
# Inference run
# -----------------------------
RUN_ROOT="$RESULTS_PARENT/$RUN_TAG"
CACHE_ROOT="$RUN_ROOT/inference_cache"
PGAS_OUTPUT_ROOT="$RUN_ROOT/pgas_output"
EVAL_ROOT="$RUN_ROOT/cli_evaluation"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$PGAS_OUTPUT_ROOT" "$RUN_ROOT/slurm"

RESOLVED_DATASET_TAGS=("${DATASET_TAGS[@]}")
if [[ ${#RESOLVED_DATASET_TAGS[@]} -eq 0 && -n "${TRIAL_SELECTION_PATH}" ]]; then
  if [[ ! -f "$TRIAL_SELECTION_PATH" ]]; then
    echo "[error] TRIAL_SELECTION_PATH not found: $TRIAL_SELECTION_PATH" >&2
    exit 7
  fi
  mapfile -t RESOLVED_DATASET_TAGS < <(
    python - "$TRIAL_SELECTION_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("Trial selection JSON must be an object: {dataset_stem: [trial_idx, ...]}")
keys = sorted(k.strip() for k in payload.keys() if str(k).strip())
for key in keys:
    print(key)
PY
  )
  if [[ ${#RESOLVED_DATASET_TAGS[@]} -eq 0 ]]; then
    echo "[error] No dataset keys found in trial selection file: $TRIAL_SELECTION_PATH" >&2
    exit 8
  fi
  echo "[setup] Resolved ${#RESOLVED_DATASET_TAGS[@]} dataset tag(s) from trial selection: $TRIAL_SELECTION_PATH"
fi

CMD=(
  python -m c_spikes.cli.run
  --data-root "$DATA_ROOT"
  --method pgas
  --smoothing-level raw
  --pgas-constants "$PGAS_CONSTANTS"
  --pgas-gparam "$PGAS_GPARAM"
  --pgas-output-root "$PGAS_OUTPUT_ROOT"
  --output-root "$EVAL_ROOT"
  --cache-root "$CACHE_ROOT"
  --run-tag "$RUN_TAG"
  --pgas-bm-sigma "$BM_SIGMA"
  --bm-sigma-spike-gap "$BM_SIGMA_GAP_S"
)

for dataset_tag in "${RESOLVED_DATASET_TAGS[@]}"; do
  CMD+=(--dataset "$dataset_tag")
done

if [[ "$USE_CACHE" == "1" ]]; then
  CMD+=(--use-cache)
fi

if [[ -n "${TRIAL_SELECTION_PATH}" ]]; then
  CMD+=(--trial-selection-path "$TRIAL_SELECTION_PATH")
fi

if [[ -n "${EDGES_PATH}" ]]; then
  CMD+=(--edges-path "$EDGES_PATH")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

# Optional cleanup:
# git -C "$REPO_ROOT" worktree remove --force "$BUILD_DIR"
