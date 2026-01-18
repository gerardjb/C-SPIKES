#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sweep synthetic spike-generation params, train ENS2 models, and evaluate them
ENS2-only on a directory of Janelia datasets (no PGAS/CASCADE, no ens2_published
per-iteration).

Usage:
  scripts/sweep_pgas_to_ens2.sh --run <pgas_run> \
                               [--cell <cell_tag> ...] \
                               [--cell-set <cell1,cell2,...> ...] \
                               [--dataset-dir <dir>] [--edges-file <npy>] [--model-root <dir>] [--eval-root <dir>] \
                               [--downsample <hz_or_raw>] [--downsample <hz_or_raw> ...] [--eval-existing] \
                               [--seed-start <int>] [--seed-stride <int>] [--corr-sigma-ms <float>] \
                               [--force-train] [--force-eval] [--force] \
                               [--dry-run] [-- <extra demo_pgas_to_ens2.py args>]

Example:
  scripts/sweep_pgas_to_ens2.sh \
    --run test_refactor \
    --cell-set jGCaMP8f_ANM471993_cell01_pgas_new_sraw_ms3_rs120_bm0p05_trial0,jGCaMP8f_ANM471993_cell02_pgas_new_sraw_ms3_rs120_bm0p05_trial0 \
    --dataset-dir data/janelia_8f/excitatory \
    -- --burnin 100

Notes:
  - Rates/smooth/duty grids are hardcoded below; edit as needed.
  - Training sweep runs once per cell set:
    - If any --cell-set is provided: each --cell-set defines one training run (comma-separated list).
    - Otherwise: all --cell values are treated as a single set.
  - For each set with K cells, sets noise_fraction = 1/K.
  - Uses deterministic per-cell noise seeds via --noise-seed-base (base advances each model).
  - Resume behavior: if checkpoint + eval outputs exist, skip the iteration.
  - Downsample eval: results are written under <eval-root>/<downsample_tag>/<model_name>, where
    downsample_tag is 'raw' or e.g. '30Hz'. Use --downsample to add additional inference rates.
  - --eval-existing skips training and evaluates any existing sweep models found in --model-root.
  - Any args after `--` are passed through to demo_pgas_to_ens2.py.
EOF
}

RUN=""
CELLS=()
CELL_SETS=()
DATASET_DIR=""
EDGES_FILE=""
MODEL_ROOT=""
EVAL_ROOT=""
SEED_START=0
SEED_STRIDE=1000
CORR_SIGMA_MS=50.0
FORCE_TRAIN=0
FORCE_EVAL=0
EVAL_EXISTING=0
DRY_RUN=0
DOWNSAMPLES=()
EXTRA_ARGS=()

CELLS_SEEN=0
CELL_SETS_SEEN=0
DOWNSAMPLES_SEEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN="$2"; shift 2 ;;
    --cell)
      if [[ "$CELLS_SEEN" -eq 0 ]]; then
        CELLS=()
        CELL_SETS=()
        CELLS_SEEN=1
      fi
      CELLS+=("$2"); shift 2 ;;
    --cell-set)
      if [[ "$CELL_SETS_SEEN" -eq 0 ]]; then
        CELL_SETS=()
        CELLS=()
        CELL_SETS_SEEN=1
      fi
      CELL_SETS+=("$2"); shift 2 ;;
    --dataset-dir)
      DATASET_DIR="$2"; shift 2 ;;
    --edges-file)
      EDGES_FILE="$2"; shift 2 ;;
    --model-root)
      MODEL_ROOT="$2"; shift 2 ;;
    --eval-root)
      EVAL_ROOT="$2"; shift 2 ;;
    --downsample)
      if [[ "$DOWNSAMPLES_SEEN" -eq 0 ]]; then
        DOWNSAMPLES=()
        DOWNSAMPLES_SEEN=1
      fi
      DOWNSAMPLES+=("$2"); shift 2 ;;
    --eval-existing|--eval-only)
      EVAL_EXISTING=1; shift 1 ;;
    --seed-start)
      SEED_START="$2"; shift 2 ;;
    --seed-stride)
      SEED_STRIDE="$2"; shift 2 ;;
    --corr-sigma-ms)
      CORR_SIGMA_MS="$2"; shift 2 ;;
    --force-train)
      FORCE_TRAIN=1; shift 1 ;;
    --force-eval)
      FORCE_EVAL=1; shift 1 ;;
    --force)
      FORCE_TRAIN=1; FORCE_EVAL=1; shift 1 ;;
    --dry-run)
      DRY_RUN=1; shift 1 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

if [[ -z "$RUN" ]]; then
  echo "ERROR: --run is required." >&2
  usage
  exit 1
fi

if [[ "$EVAL_EXISTING" -eq 0 && "$CELL_SETS_SEEN" -eq 1 && "$CELLS_SEEN" -eq 1 ]]; then
  echo "ERROR: use either --cell or --cell-set (not both)." >&2
  usage
  exit 1
fi

if [[ "$EVAL_EXISTING" -eq 0 && ${#CELL_SETS[@]} -lt 1 && ${#CELLS[@]} -lt 1 ]]; then
  echo "ERROR: at least one --cell or --cell-set is required unless --eval-existing is set." >&2
  usage
  exit 1
fi

if [[ -z "$EVAL_ROOT" ]]; then
  EVAL_ROOT="results/ens2_sweep_eval/${RUN}__excitatory"
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: dataset dir not found: $DATASET_DIR" >&2
  exit 1
fi

if [[ -n "$EDGES_FILE" && ! -f "$EDGES_FILE" ]]; then
  echo "ERROR: edges file not found: $EDGES_FILE" >&2
  exit 1
fi

if [[ "$EVAL_EXISTING" -eq 0 ]]; then
  if [[ ${#CELL_SETS[@]} -gt 0 ]]; then
    for cell_set in "${CELL_SETS[@]}"; do
      IFS=',' read -r -a cells_in_set <<< "${cell_set}"
      for cell in "${cells_in_set[@]}"; do
        ps="results/pgas_output/${RUN}/param_samples_${cell}.dat"
        if [[ ! -f "$ps" ]]; then
          echo "ERROR: param_samples not found: $ps" >&2
          exit 1
        fi
      done
    done
  else
    for cell in "${CELLS[@]}"; do
      ps="results/pgas_output/${RUN}/param_samples_${cell}.dat"
      if [[ ! -f "$ps" ]]; then
        echo "ERROR: param_samples not found: $ps" >&2
        exit 1
      fi
    done
  fi
fi

rates=(6 9 12)
smooths=(1.3 2.0)
duties=(0.35 0.45)

iter=0

EDGES_ARGS=()
if [[ -n "$EDGES_FILE" ]]; then
  EDGES_ARGS=(--edges-file "$EDGES_FILE")
fi

format_downsample_dir() {
  local ds="$1"
  if [[ "$ds" == "raw" || "$ds" == "RAW" ]]; then
    echo "raw"
    return 0
  fi
  python - <<PY
ds=float("${ds}")
if abs(ds-round(ds)) < 1e-9:
    print(f"{int(round(ds))}Hz")
else:
    s=f"{ds:.1f}".rstrip('0').rstrip('.')
    print(s.replace('.', 'p') + "Hz")
PY
}

eval_model() {
  local model_name="$1"
  local model_dir="$2"
  local ds="$3"
  local force_eval="${4:-0}"
  local ds_dir
  ds_dir="$(format_downsample_dir "$ds")"

  local eval_out="${EVAL_ROOT}/${ds_dir}/${model_name}"
  local eval_summary_json="${eval_out}/summary.json"
  local eval_summary_csv="${eval_out}/summary.csv"

  if [[ -f "$eval_summary_json" && -f "$eval_summary_csv" && "$FORCE_EVAL" -eq 0 && "$force_eval" -eq 0 ]]; then
    echo "[skip] eval outputs exist: ${eval_summary_json}"
    return 0
  fi

  local eval_cmd=(
    python scripts/eval_ens2_dir.py
    --ens2-root "${model_dir}"
    --dataset-dir "${DATASET_DIR}"
    --corr-sigma-ms "${CORR_SIGMA_MS}"
    --no-cache
    "${EDGES_ARGS[@]}"
    --out-dir "${eval_out}"
  )
  if [[ "$ds_dir" != "raw" ]]; then
    eval_cmd+=(--smoothing "${ds}")
  fi

  echo "[eval]  ${eval_cmd[*]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${eval_cmd[@]}"
  fi
}

if [[ "$EVAL_EXISTING" -eq 1 ]]; then
  mapfile -t MODEL_DIRS < <(
    find "${MODEL_ROOT}" -maxdepth 1 -type d -name "ens2_synth_${RUN}_*k*_r*_s*_d*_sb*" | sort
  )
  if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no matching models found under ${MODEL_ROOT} for run '${RUN}'." >&2
    exit 1
  fi

  echo "==> Found ${#MODEL_DIRS[@]} existing sweep models under ${MODEL_ROOT}"
  for ds in "${DOWNSAMPLES[@]}"; do
    ds_dir="$(format_downsample_dir "$ds")"
    echo "==> Downsample eval: ${ds_dir}"
    for model_dir in "${MODEL_DIRS[@]}"; do
      model_name="$(basename "${model_dir}")"
      checkpoint_path="${model_dir}/exc_ens2_pub.pt"
      if [[ ! -f "$checkpoint_path" ]]; then
        echo "[skip] missing checkpoint: ${checkpoint_path}"
        continue
      fi
      eval_model "${model_name}" "${model_dir}" "${ds}" 0
    done
  done
else
  TRAIN_CELL_SETS=()
  if [[ ${#CELL_SETS[@]} -gt 0 ]]; then
    TRAIN_CELL_SETS=("${CELL_SETS[@]}")
  else
    TRAIN_CELL_SETS=("$(IFS=','; echo "${CELLS[*]}")")
  fi

  nsets="${#TRAIN_CELL_SETS[@]}"
  for ((set_idx=0; set_idx<nsets; set_idx++)); do
    cell_set="${TRAIN_CELL_SETS[$set_idx]}"
    IFS=',' read -r -a cells_in_set <<< "${cell_set}"
    k="${#cells_in_set[@]}"
    if [[ "$k" -lt 1 ]]; then
      echo "ERROR: empty cell set at index ${set_idx}." >&2
      exit 1
    fi

    cellset_tag="cs${set_idx}"
    noise_fraction="$(awk -v kk="$k" 'BEGIN { printf "%.8f", 1.0/kk }')"
    echo "==> Cell set $((set_idx+1))/${nsets} (${cellset_tag}) K=${k} noise_fraction=${noise_fraction}"

    PARAM_ARGS=()
    for cell in "${cells_in_set[@]}"; do
      PARAM_ARGS+=(--param-samples "results/pgas_output/${RUN}/param_samples_${cell}.dat")
    done

    for rate in "${rates[@]}"; do
      for smooth in "${smooths[@]}"; do
        for duty in "${duties[@]}"; do
          smooth_tag="${smooth/./p}"
          duty_tag="${duty/./p}"
          model_tag="${cellset_tag}_k${k}_r${rate}_s${smooth_tag}_d${duty_tag}"

          # Deterministic seed schedule:
          #   - Each model uses a unique base seed (seed_start + iter*seed_stride)
          #   - demo_pgas_to_ens2.py expands this per cell: base + cell_index
          seed_base=$((SEED_START + iter*SEED_STRIDE))
          iter=$((iter+1))

          run_tag="${model_tag}_sb${seed_base}"
          model_name="ens2_synth_${RUN}_${run_tag}"

          echo "==> model=${model_name} spike_rate=${rate} spike_params=(${smooth},${duty}) seed_base=${seed_base}"

          model_dir="${MODEL_ROOT}/${model_name}"
          checkpoint_path="${model_dir}/exc_ens2_pub.pt"

          need_train=0
          if [[ "$FORCE_TRAIN" -eq 1 || ! -f "$checkpoint_path" ]]; then
            need_train=1
          fi

          train_cmd=(
            python scripts/demo_pgas_to_ens2.py
            "${PARAM_ARGS[@]}"
            --spike-rate "${rate}"
            --spike-params "${smooth}" "${duty}"
            --noise-fraction "${noise_fraction}"
            --noise-seed-base "${seed_base}"
            --synth-tag-suffix "${run_tag}"
            --model-name "${model_name}"
            --model-root "${MODEL_ROOT}"
            --run-tag "${run_tag}"
            --train-ens2
            "${EXTRA_ARGS[@]}"
          )
          if [[ "$need_train" -eq 1 ]]; then
            echo "[train] ${train_cmd[*]}"
            if [[ "$DRY_RUN" -eq 0 ]]; then
              "${train_cmd[@]}"
            fi
          else
            echo "[skip] checkpoint exists: ${checkpoint_path}"
          fi

          for ds in "${DOWNSAMPLES[@]}"; do
            eval_model "${model_name}" "${model_dir}" "${ds}" "${need_train}"
          done
        done
      done
    done
  done
fi

# Baseline: evaluate the stock/published model once per downsample setting (ENS2-only).
for ds in "${DOWNSAMPLES[@]}"; do
  ds_dir="$(format_downsample_dir "$ds")"
  baseline_out="${EVAL_ROOT}/${ds_dir}/baseline_ens2_published"
  baseline_summary_json="${baseline_out}/summary.json"
  baseline_summary_csv="${baseline_out}/summary.csv"
  baseline_cmd=(
    python scripts/eval_ens2_dir.py
    --ens2-root "results/Pretrained_models/ens2_published"
    --dataset-dir "${DATASET_DIR}"
    --corr-sigma-ms "${CORR_SIGMA_MS}"
    --no-cache
    "${EDGES_ARGS[@]}"
    --out-dir "${baseline_out}"
  )
  if [[ "$ds_dir" != "raw" ]]; then
    baseline_cmd+=(--smoothing "${ds}")
  fi
  echo "[baseline] ${baseline_cmd[*]}"
  if [[ -f "$baseline_summary_json" && -f "$baseline_summary_csv" && "$FORCE_EVAL" -eq 0 ]]; then
    echo "[skip] baseline eval outputs exist: ${baseline_summary_json}"
  else
    if [[ "$DRY_RUN" -eq 0 ]]; then
      "${baseline_cmd[@]}"
    fi
  fi
done
