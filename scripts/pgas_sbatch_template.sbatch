#!/bin/bash
#SBATCH --job-name=pgas_allan_commit
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pgas_allan_commit_%j.out
#SBATCH --error=logs/pgas_allan_commit_%j.err

set -eo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <run_tag> <commit_hash> [feature_flags]" >&2
  exit 2
fi

RUN_TAG="$1"
COMMIT_HASH="$2"
FEATURE_FLAGS="${3:-}"
COMMIT_SHORT="$(echo "$COMMIT_HASH" | cut -c1-8)"
ENV_NAME="${ENV_NAME:-${RUN_TAG}_build}"

feature_enabled() {
  local feature="$1"
  [[ ",${FEATURE_FLAGS}," == *",${feature},"* ]]
}

bool_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

verify_pgas_import() {
  python - <<'PY'
import c_spikes
import c_spikes.pgas.pgas_bound as pgas_bound

print(f"[setup] c_spikes imported from: {c_spikes.__file__}")
print(f"[setup] PGAS backend selected: {getattr(pgas_bound, '__backend__', 'unknown')}")
PY
}

write_first_trial_selection() {
  local output_path="$1"
  shift
  python - "$output_path" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
dataset_tags = [tag for tag in sys.argv[2:] if tag]
payload = {tag: [0] for tag in dataset_tags}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -e "$SLURM_SUBMIT_DIR/.git" ]]; then
  DEFAULT_REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Override these through sbatch --export or by editing a local copy of this template.
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/sample_data/janelia_8f/excitatory}"
RESULTS_PARENT="${RESULTS_PARENT:-$DATA_ROOT/spike_inference}"
PGAS_CONSTANTS="${PGAS_CONSTANTS:-$REPO_ROOT/parameter_files/constants_GCaMP8_soma.json}"
PGAS_GPARAM="${PGAS_GPARAM:-$REPO_ROOT/src/c_spikes/pgas/20230525_gold.dat}"
EDGES_PATH="${EDGES_PATH:-}"
TRIAL_SELECTION_PATH="${TRIAL_SELECTION_PATH:-}"
EFFECTIVE_TRIAL_SELECTION_PATH="$TRIAL_SELECTION_PATH"
DATASET_TAGS="${DATASET_TAGS:-}"
SMOOTHING_LEVEL="${SMOOTHING_LEVEL:-raw}"
USE_CACHE="${USE_CACHE:-0}"
FIRST_TRIAL_ONLY="${FIRST_TRIAL_ONLY:-0}"
TRIALWISE_CORRELATIONS="${TRIALWISE_CORRELATIONS:-1}"
CORR_SIGMA_MS="${CORR_SIGMA_MS:-50}"

# Calibration defaults are intentionally suitable for the post-Kalman Allan branch.
# Older commits only receive flags they support, based on FEATURE_FLAGS.
BM_SIGMA="${BM_SIGMA:-auto}"
BM_SIGMA_MIN="${BM_SIGMA_MIN:-0.0005}"
BM_SIGMA_MAX="${BM_SIGMA_MAX:-0.5}"
BM_SIGMA_GAP_S="${BM_SIGMA_GAP_S:-0.15}"
PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK="${PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK:-0}"
PGAS_SIGMA2_TARGET="${PGAS_SIGMA2_TARGET:-}"
PGAS_SIGMA2_ALPHA="${PGAS_SIGMA2_ALPHA:-}"
PGAS_SIGMA2_PRIOR_STRENGTH="${PGAS_SIGMA2_PRIOR_STRENGTH:-4}"
PGAS_RESAMPLE_FS="${PGAS_RESAMPLE_FS:-}"
PGAS_MAXSPIKES="${PGAS_MAXSPIKES:-}"
PGAS_C0_FIRST_Y="${PGAS_C0_FIRST_Y:-0}"
C_SPIKES_PGAS_BACKEND="${C_SPIKES_PGAS_BACKEND:-}"
SWEEP_JSON="${SWEEP_JSON:-}"
SWEEP_BASE_RUN_TAG="${SWEEP_BASE_RUN_TAG:-}"
SWEEP_INCLUDE_CONFIG_TAGS="${SWEEP_INCLUDE_CONFIG_TAGS:-}"
SWEEP_EXCLUDE_CONFIG_TAGS="${SWEEP_EXCLUDE_CONFIG_TAGS:-}"
SWEEP_CONTINUE_ON_ERROR="${SWEEP_CONTINUE_ON_ERROR:-0}"
SWEEP_DRY_RUN="${SWEEP_DRY_RUN:-0}"

CONDA_BASE_ENV="${CONDA_BASE_ENV:-c_spikes_e}"
CONDA_OFFLINE_MODE="${CONDA_OFFLINE_MODE:-1}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2024.6}"
CUDATOOLKIT_MODULE="${CUDATOOLKIT_MODULE:-cudatoolkit/12.9}"
CMAKE_MODULE="${CMAKE_MODULE:-cmake/3.30.8}"
KOKKOS_SOURCE_OVERRIDE="${KOKKOS_SOURCE_OVERRIDE:-}"
PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS:---no-deps --no-build-isolation}"
REFRESH_CONSTANTS_CACHE="${REFRESH_CONSTANTS_CACHE:-1}"
CLEAN_BUILD_WORKTREE="${CLEAN_BUILD_WORKTREE:-0}"
REUSE_EXISTING_BUILD="${REUSE_EXISTING_BUILD:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

BUILD_PARENT="${BUILD_PARENT:-$RESULTS_PARENT/_builds}"
RUN_ROOT="$RESULTS_PARENT/$RUN_TAG"
CACHE_ROOT="$RUN_ROOT/inference_cache"
PGAS_OUTPUT_ROOT="$RUN_ROOT/pgas_output"
EVAL_ROOT="$RUN_ROOT/cli_evaluation"
BUILD_DIR="$BUILD_PARENT/${RUN_TAG}_${COMMIT_SHORT}"
REQUESTED_BUILD_DIR="$BUILD_DIR"
REQUESTED_ENV_NAME="$ENV_NAME"
BUILD_REUSED=0

echo "[setup] run_tag=$RUN_TAG commit=$COMMIT_HASH features=${FEATURE_FLAGS:-none}"
echo "[setup] script_dir=$SCRIPT_DIR"
echo "[setup] slurm_submit_dir=${SLURM_SUBMIT_DIR:-}"
echo "[setup] repo_root=$REPO_ROOT"
echo "[setup] results_parent=$RESULTS_PARENT"

if [[ "${PGAS_TEMPLATE_TEST_RESOLVE:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${PGAS_TEMPLATE_TEST_FIRST_TRIAL:-0}" == "1" ]]; then
  mkdir -p "$RUN_ROOT/slurm"
  TEST_DATASET_TAGS=()
  if [[ -n "$DATASET_TAGS" ]]; then
    read -r -a TEST_DATASET_TAGS <<< "$DATASET_TAGS"
  fi
  if [[ ${#TEST_DATASET_TAGS[@]} -eq 0 ]]; then
    echo "[error] Set DATASET_TAGS for PGAS_TEMPLATE_TEST_FIRST_TRIAL." >&2
    exit 9
  fi
  if bool_enabled "$FIRST_TRIAL_ONLY"; then
    EFFECTIVE_TRIAL_SELECTION_PATH="$RUN_ROOT/slurm/first_trial_selection.json"
    write_first_trial_selection "$EFFECTIVE_TRIAL_SELECTION_PATH" "${TEST_DATASET_TAGS[@]}"
    echo "[setup] FIRST_TRIAL_ONLY generated trial selection: $EFFECTIVE_TRIAL_SELECTION_PATH"
    cat "$EFFECTIVE_TRIAL_SELECTION_PATH"
  else
    echo "[setup] FIRST_TRIAL_ONLY is not enabled."
  fi
  exit 0
fi

source /etc/profile.d/modules.sh
module load "$ANACONDA_MODULE"
if [[ -n "$CUDATOOLKIT_MODULE" ]]; then
  module load "$CUDATOOLKIT_MODULE"
fi
if [[ -n "$CMAKE_MODULE" ]]; then
  module load "$CMAKE_MODULE"
fi

if [[ "$CONDA_OFFLINE_MODE" == "1" ]]; then
  export CONDA_OFFLINE=true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda command not found after loading $ANACONDA_MODULE" >&2
  exit 3
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

mkdir -p "$BUILD_PARENT" "$RUN_ROOT/slurm" "$CACHE_ROOT" "$PGAS_OUTPUT_ROOT" "$EVAL_ROOT"

if [[ "$REFRESH_CONSTANTS_CACHE" == "1" ]]; then
  rm -rf "$CACHE_ROOT/pgas_constants"
fi

if bool_enabled "$REUSE_EXISTING_BUILD" && ! bool_enabled "$FORCE_REBUILD"; then
  EXPECTED_COMMIT="$(git -C "$REPO_ROOT" rev-parse "${COMMIT_HASH}^{commit}" 2>/dev/null || true)"
  REUSE_CANDIDATES=()
  if [[ -d "$REQUESTED_BUILD_DIR" ]]; then
    REUSE_CANDIDATES+=("$REQUESTED_BUILD_DIR")
  fi
  while IFS= read -r candidate; do
    REUSE_CANDIDATES+=("$candidate")
  done < <(find "$BUILD_PARENT" -mindepth 1 -maxdepth 1 -type d -name "*_${COMMIT_SHORT}" 2>/dev/null | sort)

  SEEN_REUSE_CANDIDATES="|"
  for candidate in "${REUSE_CANDIDATES[@]}"; do
    if [[ "$SEEN_REUSE_CANDIDATES" == *"|$candidate|"* ]]; then
      continue
    fi
    SEEN_REUSE_CANDIDATES="${SEEN_REUSE_CANDIDATES}${candidate}|"
    if [[ ! -e "$candidate/.git" ]]; then
      continue
    fi
    CANDIDATE_COMMIT="$(git -C "$candidate" rev-parse HEAD 2>/dev/null || true)"
    if [[ -z "$EXPECTED_COMMIT" || "$CANDIDATE_COMMIT" != "$EXPECTED_COMMIT" ]]; then
      continue
    fi
    CANDIDATE_BASE="$(basename "$candidate")"
    CANDIDATE_RUN_TAG="${CANDIDATE_BASE%_${COMMIT_SHORT}}"
    CANDIDATE_ENV_NAME="${CANDIDATE_RUN_TAG}_build"
    if conda_env_exists "$CANDIDATE_ENV_NAME"; then
      BUILD_DIR="$candidate"
      ENV_NAME="$CANDIDATE_ENV_NAME"
      BUILD_REUSED=1
      echo "[setup] Reusing existing build: build_dir=$BUILD_DIR env=$ENV_NAME"
      break
    fi
    echo "[setup] Reuse candidate has matching worktree but no env: $candidate env=$CANDIDATE_ENV_NAME"
  done
fi

if [[ "${PGAS_TEMPLATE_TEST_REUSE_BUILD:-0}" == "1" ]]; then
  echo "[test] build_reused=$BUILD_REUSED"
  echo "[test] build_dir=$BUILD_DIR"
  echo "[test] env_name=$ENV_NAME"
  exit 0
fi

if [[ "$BUILD_REUSED" == "1" ]]; then
  conda activate "$ENV_NAME"
  unset PYTHONPATH
  if [[ -n "$C_SPIKES_PGAS_BACKEND" ]]; then
    export C_SPIKES_PGAS_BACKEND
  fi
  if ! verify_pgas_import; then
    echo "[warn] Reused build failed PGAS import check; rebuilding requested build." >&2
    conda deactivate || true
    BUILD_REUSED=0
    BUILD_DIR="$REQUESTED_BUILD_DIR"
    ENV_NAME="$REQUESTED_ENV_NAME"
  fi
fi

if [[ "$BUILD_REUSED" != "1" ]]; then
  if conda_env_exists "$ENV_NAME"; then
    conda env remove -y -n "$ENV_NAME"
  fi
  if ! conda_env_exists "$CONDA_BASE_ENV"; then
    echo "[error] Base env '$CONDA_BASE_ENV' not found. Create it first on a login node or set CONDA_BASE_ENV." >&2
    exit 4
  fi
  conda create -y -n "$ENV_NAME" --clone "$CONDA_BASE_ENV"
  conda activate "$ENV_NAME"

  if ! command -v g++ >/dev/null 2>&1; then
    echo "[error] g++ not found. Load a compiler module before build." >&2
    exit 5
  fi
  if ! command -v gfortran >/dev/null 2>&1; then
    echo "[error] gfortran not found. Load a compiler module before build." >&2
    exit 5
  fi

  git -C "$REPO_ROOT" worktree prune --expire now || true
  git -C "$REPO_ROOT" worktree remove --force "$BUILD_DIR" 2>/dev/null || true
  git -C "$REPO_ROOT" worktree prune --expire now || true
  if [[ -d "$BUILD_DIR" ]]; then
    rm -rf "$BUILD_DIR"
  fi
  git -C "$REPO_ROOT" worktree add --detach -f "$BUILD_DIR" "$COMMIT_HASH"
  git -C "$BUILD_DIR" checkout --detach "$COMMIT_HASH"

  if [[ "$CLEAN_BUILD_WORKTREE" == "1" ]]; then
    cleanup_worktree() {
      git -C "$REPO_ROOT" worktree remove --force "$BUILD_DIR" 2>/dev/null || true
    }
    trap cleanup_worktree EXIT
  fi

  if [[ ! -f "$REPO_ROOT/vcpkg/scripts/buildsystems/vcpkg.cmake" ]]; then
    echo "[error] vcpkg toolchain not found at $REPO_ROOT/vcpkg/scripts/buildsystems/vcpkg.cmake" >&2
    echo "[error] Symlink or install vcpkg under REPO_ROOT before running commit sweeps." >&2
    exit 6
  fi
  ln -sfn "$REPO_ROOT/vcpkg" "$BUILD_DIR/vcpkg"
  export VCPKG_ROOT="$BUILD_DIR/vcpkg"
  export CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$BUILD_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake${CMAKE_ARGS:+ $CMAKE_ARGS}"

  if [[ -z "$KOKKOS_SOURCE_OVERRIDE" ]]; then
    if [[ -d "$REPO_ROOT/build/_deps/kokkos-src" ]]; then
      KOKKOS_SOURCE_OVERRIDE="$REPO_ROOT/build/_deps/kokkos-src"
    elif [[ -d "$REPO_ROOT/build" ]]; then
      KOKKOS_SOURCE_OVERRIDE="$(find "$REPO_ROOT/build" -maxdepth 5 -type d -name kokkos-src 2>/dev/null | head -n 1 || true)"
    fi
  fi

  if [[ -n "$KOKKOS_SOURCE_OVERRIDE" && -d "$KOKKOS_SOURCE_OVERRIDE" ]]; then
    echo "[setup] Using offline Kokkos source: $KOKKOS_SOURCE_OVERRIDE"
    export KOKKOS_SOURCE_OVERRIDE
    export CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_KOKKOS=$KOKKOS_SOURCE_OVERRIDE -DFETCHCONTENT_SOURCE_DIR_Kokkos=$KOKKOS_SOURCE_OVERRIDE ${CMAKE_ARGS}"
  else
    echo "[error] Local Kokkos source not found: ${KOKKOS_SOURCE_OVERRIDE:-unset}" >&2
    echo "[error] Set KOKKOS_SOURCE_OVERRIDE to an existing kokkos-src path." >&2
    exit 7
  fi

  DEPS_CMAKE="$BUILD_DIR/dependencies/CMakeLists.txt"
  if [[ -f "$DEPS_CMAKE" ]] && grep -q "github.com/kokkos/kokkos.git" "$DEPS_CMAKE"; then
    echo "[setup] Patching dependencies/CMakeLists.txt to use local Kokkos source"
    sed -i "s|GIT_REPOSITORY https://github.com/kokkos/kokkos.git|SOURCE_DIR \"$KOKKOS_SOURCE_OVERRIDE\"|g" "$DEPS_CMAKE"
    sed -i '/^[[:space:]]*GIT_TAG[[:space:]]\+/d' "$DEPS_CMAKE"
  fi

  PIP_INSTALL_ARGS_ARRAY=()
  if [[ -n "$PIP_INSTALL_ARGS" ]]; then
    read -r -a PIP_INSTALL_ARGS_ARRAY <<< "$PIP_INSTALL_ARGS"
  fi
  python -m pip install "${PIP_INSTALL_ARGS_ARRAY[@]}" "$BUILD_DIR"

  unset PYTHONPATH
  if [[ -n "$C_SPIKES_PGAS_BACKEND" ]]; then
    export C_SPIKES_PGAS_BACKEND
  fi
  verify_pgas_import
fi

if [[ -n "$SWEEP_JSON" ]]; then
  SWEEP_CMD=(
    python "$REPO_ROOT/scripts/run_pgas_inference_sweep.py"
    --sweep-json "$SWEEP_JSON"
    --repo-root "$REPO_ROOT"
    --data-root "$DATA_ROOT"
    --results-parent "$RESULTS_PARENT"
    --pgas-constants "$PGAS_CONSTANTS"
    --pgas-gparam "$PGAS_GPARAM"
    --features "$FEATURE_FLAGS"
  )
  if [[ -n "$SWEEP_BASE_RUN_TAG" ]]; then
    SWEEP_CMD+=(--base-run-tag "$SWEEP_BASE_RUN_TAG")
  fi
  if [[ -n "$SWEEP_INCLUDE_CONFIG_TAGS" ]]; then
    SWEEP_INCLUDE_CONFIG_TAGS_NORMALIZED="${SWEEP_INCLUDE_CONFIG_TAGS//:/ }"
    read -r -a SWEEP_INCLUDE_CONFIG_TAGS_ARRAY <<< "$SWEEP_INCLUDE_CONFIG_TAGS_NORMALIZED"
    for config_tag in "${SWEEP_INCLUDE_CONFIG_TAGS_ARRAY[@]}"; do
      SWEEP_CMD+=(--include-config-tag "$config_tag")
    done
  fi
  if [[ -n "$SWEEP_EXCLUDE_CONFIG_TAGS" ]]; then
    SWEEP_EXCLUDE_CONFIG_TAGS_NORMALIZED="${SWEEP_EXCLUDE_CONFIG_TAGS//:/ }"
    read -r -a SWEEP_EXCLUDE_CONFIG_TAGS_ARRAY <<< "$SWEEP_EXCLUDE_CONFIG_TAGS_NORMALIZED"
    for config_tag in "${SWEEP_EXCLUDE_CONFIG_TAGS_ARRAY[@]}"; do
      SWEEP_CMD+=(--exclude-config-tag "$config_tag")
    done
  fi
  if [[ "$SWEEP_CONTINUE_ON_ERROR" == "1" ]]; then
    SWEEP_CMD+=(--continue-on-error)
  fi
  if [[ "$SWEEP_DRY_RUN" == "1" ]]; then
    SWEEP_CMD+=(--dry-run)
  fi
  printf '[sweep-run]'
  printf ' %q' "${SWEEP_CMD[@]}"
  printf '\n'
  "${SWEEP_CMD[@]}"
  exit
fi

RESOLVED_DATASET_TAGS=()
if [[ -n "$DATASET_TAGS" ]]; then
  read -r -a RESOLVED_DATASET_TAGS <<< "$DATASET_TAGS"
elif [[ -n "$EFFECTIVE_TRIAL_SELECTION_PATH" ]]; then
  if [[ ! -f "$EFFECTIVE_TRIAL_SELECTION_PATH" ]]; then
    echo "[error] TRIAL_SELECTION_PATH not found: $EFFECTIVE_TRIAL_SELECTION_PATH" >&2
    exit 8
  fi
  mapfile -t RESOLVED_DATASET_TAGS < <(
    python - "$EFFECTIVE_TRIAL_SELECTION_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("Trial selection JSON must be an object: {dataset_stem: [trial_idx, ...]}")
for key in sorted(str(key).strip() for key in payload.keys()):
    if key:
        print(key)
PY
  )
fi

if [[ ${#RESOLVED_DATASET_TAGS[@]} -eq 0 ]]; then
  echo "[error] Set DATASET_TAGS or TRIAL_SELECTION_PATH so the CLI knows which datasets to run." >&2
  exit 9
fi

if bool_enabled "$FIRST_TRIAL_ONLY" && [[ -z "$EFFECTIVE_TRIAL_SELECTION_PATH" ]]; then
  EFFECTIVE_TRIAL_SELECTION_PATH="$RUN_ROOT/slurm/first_trial_selection.json"
  write_first_trial_selection "$EFFECTIVE_TRIAL_SELECTION_PATH" "${RESOLVED_DATASET_TAGS[@]}"
  echo "[setup] FIRST_TRIAL_ONLY generated trial selection: $EFFECTIVE_TRIAL_SELECTION_PATH"
elif bool_enabled "$FIRST_TRIAL_ONLY"; then
  echo "[setup] FIRST_TRIAL_ONLY requested with explicit TRIAL_SELECTION_PATH; explicit selection takes precedence."
fi

MANIFEST_PATH="$RUN_ROOT/slurm/build_manifest_${COMMIT_SHORT}.json"
export RUN_TAG COMMIT_HASH FEATURE_FLAGS BUILD_DIR ENV_NAME REPO_ROOT DATA_ROOT RESULTS_PARENT
export PGAS_CONSTANTS PGAS_GPARAM EDGES_PATH TRIAL_SELECTION_PATH EFFECTIVE_TRIAL_SELECTION_PATH
export DATASET_TAGS SMOOTHING_LEVEL FIRST_TRIAL_ONLY
export BM_SIGMA BM_SIGMA_MIN BM_SIGMA_MAX BM_SIGMA_GAP_S PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK
export PGAS_SIGMA2_TARGET PGAS_SIGMA2_ALPHA PGAS_SIGMA2_PRIOR_STRENGTH PGAS_RESAMPLE_FS
python - "$MANIFEST_PATH" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

keys = [
    "RUN_TAG",
    "COMMIT_HASH",
    "FEATURE_FLAGS",
    "BUILD_DIR",
    "ENV_NAME",
    "REPO_ROOT",
    "DATA_ROOT",
    "RESULTS_PARENT",
    "PGAS_CONSTANTS",
    "PGAS_GPARAM",
    "EDGES_PATH",
    "TRIAL_SELECTION_PATH",
    "EFFECTIVE_TRIAL_SELECTION_PATH",
    "DATASET_TAGS",
    "SMOOTHING_LEVEL",
    "FIRST_TRIAL_ONLY",
    "BM_SIGMA",
    "BM_SIGMA_MIN",
    "BM_SIGMA_MAX",
    "BM_SIGMA_GAP_S",
    "PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK",
    "PGAS_SIGMA2_TARGET",
    "PGAS_SIGMA2_ALPHA",
    "PGAS_SIGMA2_PRIOR_STRENGTH",
    "PGAS_RESAMPLE_FS",
]
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "env": {key: os.environ.get(key, "") for key in keys},
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

CMD=(
  python -m c_spikes.cli.run
  --data-root "$DATA_ROOT"
  --method pgas
  --smoothing-level "$SMOOTHING_LEVEL"
  --pgas-constants "$PGAS_CONSTANTS"
  --pgas-gparam "$PGAS_GPARAM"
  --pgas-output-root "$PGAS_OUTPUT_ROOT"
  --output-root "$EVAL_ROOT"
  --cache-root "$CACHE_ROOT"
  --run-tag "$RUN_TAG"
  --pgas-bm-sigma "$BM_SIGMA"
  --bm-sigma-spike-gap "$BM_SIGMA_GAP_S"
  --corr-sigma-ms "$CORR_SIGMA_MS"
)

if feature_enabled bm_bounds; then
  CMD+=(--pgas-bm-sigma-min "$BM_SIGMA_MIN")
  CMD+=(--pgas-bm-sigma-max "$BM_SIGMA_MAX")
elif [[ "$BM_SIGMA" == "auto" ]]; then
  echo "[warn] Commit does not expose bm_sigma bounds; auto calibration will use commit defaults."
fi

if feature_enabled low_activity_mask; then
  if bool_enabled "$PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK"; then
    CMD+=(--pgas-bm-sigma-use-low-activity-mask)
  fi
elif bool_enabled "$PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK"; then
  echo "[warn] Commit does not expose low-activity-mask; ignoring PGAS_BM_SIGMA_USE_LOW_ACTIVITY_MASK."
fi

if feature_enabled sigma2_prior; then
  if [[ -n "$PGAS_SIGMA2_TARGET" ]]; then
    CMD+=(--pgas-sigma2-target "$PGAS_SIGMA2_TARGET")
    CMD+=(--pgas-sigma2-prior-strength "$PGAS_SIGMA2_PRIOR_STRENGTH")
    if [[ -n "$PGAS_SIGMA2_ALPHA" ]]; then
      CMD+=(--pgas-sigma2-alpha "$PGAS_SIGMA2_ALPHA")
    fi
  elif [[ -n "$PGAS_SIGMA2_ALPHA" ]]; then
    CMD+=(--pgas-sigma2-alpha "$PGAS_SIGMA2_ALPHA")
  fi
elif [[ -n "$PGAS_SIGMA2_TARGET" || -n "$PGAS_SIGMA2_ALPHA" ]]; then
  echo "[warn] Commit does not expose sigma2 prior knobs; ignoring sigma2 overrides."
fi

if [[ -n "$PGAS_RESAMPLE_FS" ]]; then
  CMD+=(--pgas-resample-fs "$PGAS_RESAMPLE_FS")
fi
if [[ -n "$PGAS_MAXSPIKES" ]]; then
  CMD+=(--pgas-maxspikes "$PGAS_MAXSPIKES")
fi
if bool_enabled "$PGAS_C0_FIRST_Y"; then
  CMD+=(--pgas-c0-first-y)
fi
if bool_enabled "$FIRST_TRIAL_ONLY"; then
  CMD+=(--first-trial-only)
fi
if bool_enabled "$TRIALWISE_CORRELATIONS"; then
  CMD+=(--trialwise-correlations)
fi
if bool_enabled "$USE_CACHE"; then
  CMD+=(--use-cache)
fi
if [[ -n "$EFFECTIVE_TRIAL_SELECTION_PATH" ]]; then
  CMD+=(--trial-selection-path "$EFFECTIVE_TRIAL_SELECTION_PATH")
fi
if [[ -n "$EDGES_PATH" ]]; then
  CMD+=(--edges-path "$EDGES_PATH")
fi

for dataset_tag in "${RESOLVED_DATASET_TAGS[@]}"; do
  CMD+=(--dataset "$dataset_tag")
done

printf '[run]'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
