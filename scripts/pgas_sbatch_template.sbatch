#!/bin/bash
#SBATCH --job-name=pgas_commit_eval
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=<your_path_to_slurm_logs>/%x_%j.out
#SBATCH --error=<your_path_to_slurm_logs>/%x_%j.err

set -euo pipefail

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
REPO_ROOT="<your_path_to_repo_root>"
DATA_ROOT="<your_path_to_data_root>"
RESULTS_PARENT="<your_path_to_data_root>/spike_inference"
SLURM_LOG_DIR="<your_path_to_slurm_logs>"
PGAS_CONSTANTS="$REPO_ROOT/parameter_files/constants_GCaMP8_soma.json"
PGAS_GPARAM="$REPO_ROOT/src/c_spikes/pgas/20230525_gold.dat"
EDGES_PATH="<your_path_to_edges_npy_or_leave_empty>"
TRIAL_SELECTION_PATH="<your_path_to_trial_selection_json_or_leave_empty>"
DATASET_TAGS=("jGCaMP8f_ANM471993_cell01")
USE_CACHE=1
BM_SIGMA="0.02"
BM_SIGMA_GAP_S="0.15"
PYTHON_VERSION="3.11"

# -----------------------------
# Environment + build setup
# -----------------------------
source /etc/profile.d/modules.sh
module load anaconda3/2024.6

mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$RESULTS_PARENT/_builds"

BUILD_DIR="$RESULTS_PARENT/_builds/${RUN_TAG}_${COMMIT_SHORT}"

echo "[setup] run_tag=$RUN_TAG commit=$COMMIT_HASH env=$ENV_NAME"

# Fresh env per run/build name for clean, non-editable installs.
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env remove -y -n "$ENV_NAME"
fi
conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# Use a detached worktree so main branch state is untouched.
if [[ -d "$BUILD_DIR" ]]; then
  rm -rf "$BUILD_DIR"
fi
git -C "$REPO_ROOT" worktree add --detach "$BUILD_DIR" "$COMMIT_HASH"
git -C "$BUILD_DIR" checkout --detach "$COMMIT_HASH"

# Non-editable install from the commit snapshot.
python -m pip install "$BUILD_DIR"

# -----------------------------
# Inference run
# -----------------------------
RUN_ROOT="$RESULTS_PARENT/$RUN_TAG"
CACHE_ROOT="$RUN_ROOT/inference_cache"
PGAS_OUTPUT_ROOT="$RUN_ROOT/pgas_output"
EVAL_ROOT="$RUN_ROOT/cli_evaluation"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$PGAS_OUTPUT_ROOT" "$RUN_ROOT/slurm"

CMD=(
  python "$BUILD_DIR/run_pipeline.py"
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

for dataset_tag in "${DATASET_TAGS[@]}"; do
  CMD+=(--dataset "$dataset_tag")
done

if [[ "$USE_CACHE" == "1" ]]; then
  CMD+=(--use-cache)
fi

if [[ -n "${TRIAL_SELECTION_PATH}" && "${TRIAL_SELECTION_PATH}" != "<your_path_to_trial_selection_json_or_leave_empty>" ]]; then
  CMD+=(--trial-selection-path "$TRIAL_SELECTION_PATH")
fi

if [[ -n "${EDGES_PATH}" && "${EDGES_PATH}" != "<your_path_to_edges_npy_or_leave_empty>" ]]; then
  CMD+=(--edges-path "$EDGES_PATH")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

# Optional cleanup:
# git -C "$REPO_ROOT" worktree remove --force "$BUILD_DIR"

