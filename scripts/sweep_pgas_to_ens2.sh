#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sweep synthetic spike-generation params, train ENS2 models, and evaluate them
ENS2-only on a directory of Janelia datasets (no PGAS/CASCADE, no ens2_published
per-iteration).

Usage:
  scripts/sweep_pgas_to_ens2.sh --run <pgas_run> --cell <cell_tag> [--cell <cell_tag> ...] \
                               [--dataset-dir <dir>] [--edges-file <npy>] [--model-root <dir>] [--eval-root <dir>] \
                               [--seed-start <int>] [--seed-stride <int>] [--corr-sigma-ms <float>] \
                               [--force-train] [--force-eval] [--force] \
                               [--dry-run] [-- <extra demo_pgas_to_ens2.py args>]

Example:
  scripts/sweep_pgas_to_ens2.sh \
    --run test_refactor \
    --cell jGCaMP8f_ANM471993_cell01_pgas_new_sraw_ms3_rs120_bm0p05_trial0 \
    --cell jGCaMP8f_ANM471993_cell02_pgas_new_sraw_ms3_rs120_bm0p05_trial0 \
    --dataset-dir data/janelia_8f/excitatory \
    -- --burnin 100

Notes:
  - Rates/smooth/duty grids are hardcoded below; edit as needed.
  - For each K=1..N cells, trains on the first K cells provided.
  - For each K, sets noise_fraction = 1/K.
  - Uses deterministic per-cell noise seeds via --noise-seed-base (base advances each model).
  - Resume behavior: if checkpoint + eval outputs exist, skip the iteration.
  - Any args after `--` are passed through to demo_pgas_to_ens2.py.
EOF
}

RUN="comparison"
CELLS=(
  "jGCaMP8f_ANM478349_cell04_ms2_trial1"
  "jGCaMP8f_ANM478349_cell01_ms2_trial3"
  "jGCaMP8f_ANM478407_cell01_ms2_trial3"
  "jGCaMP8f_ANM478411_cell02_ms2_trial1"
  "jGCaMP8f_ANM471994_cell05_ms2_trial1"
)
DATASET_DIR="data/janelia_8f/excitatory"
EDGES_FILE=""
MODEL_ROOT="results/Pretrained_models"
EVAL_ROOT=""
SEED_START=0
SEED_STRIDE=1000
CORR_SIGMA_MS=50.0
FORCE_TRAIN=0
FORCE_EVAL=0
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN="$2"; shift 2 ;;
    --cell)
      CELLS+=("$2"); shift 2 ;;
    --dataset-dir)
      DATASET_DIR="$2"; shift 2 ;;
    --edges-file)
      EDGES_FILE="$2"; shift 2 ;;
    --model-root)
      MODEL_ROOT="$2"; shift 2 ;;
    --eval-root)
      EVAL_ROOT="$2"; shift 2 ;;
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
      EXTRA_ARGS+=("$@")
      break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

if [[ -z "$RUN" || ${#CELLS[@]} -lt 1 ]]; then
  echo "ERROR: --run and at least one --cell are required." >&2
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

for cell in "${CELLS[@]}"; do
  ps="results/pgas_output/${RUN}/param_samples_${cell}.dat"
  if [[ ! -f "$ps" ]]; then
    echo "ERROR: param_samples not found: $ps" >&2
    exit 1
  fi
done

rates=(6 9 12)
smooths=(1.3 2.0)
duties=(0.35 0.45)

iter=0
ncells="${#CELLS[@]}"

EDGES_ARGS=()
if [[ -n "$EDGES_FILE" ]]; then
  EDGES_ARGS=(--edges-file "$EDGES_FILE")
fi

for ((k=1; k<=ncells; k++)); do
  noise_fraction="$(awk -v kk="$k" 'BEGIN { printf "%.8f", 1.0/kk }')"
  echo "==> K=${k}/${ncells} cells: noise_fraction=${noise_fraction}"

  PARAM_ARGS=()
  for ((i=0; i<k; i++)); do
    cell="${CELLS[$i]}"
    PARAM_ARGS+=(--param-samples "results/pgas_output/${RUN}/param_samples_${cell}.dat")
  done

  for rate in "${rates[@]}"; do
    for smooth in "${smooths[@]}"; do
      for duty in "${duties[@]}"; do
        smooth_tag="${smooth/./p}"
        duty_tag="${duty/./p}"
        model_tag="k${k}_r${rate}_s${smooth_tag}_d${duty_tag}"

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
        eval_out="${EVAL_ROOT}/${model_name}"
        eval_summary_json="${eval_out}/summary.json"
        eval_summary_csv="${eval_out}/summary.csv"

        need_train=0
        if [[ "$FORCE_TRAIN" -eq 1 || ! -f "$checkpoint_path" ]]; then
          need_train=1
        fi

        need_eval=0
        if [[ "$FORCE_EVAL" -eq 1 || "$need_train" -eq 1 || ! -f "$eval_summary_json" || ! -f "$eval_summary_csv" ]]; then
          need_eval=1
        fi

        if [[ "$need_train" -eq 0 && "$need_eval" -eq 0 ]]; then
          echo "[skip] already trained+evaluated: ${model_name}"
          continue
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

        eval_cmd=(
          python scripts/eval_ens2_dir.py
          --ens2-root "${MODEL_ROOT}/${model_name}"
          --dataset-dir "${DATASET_DIR}"
          --corr-sigma-ms "${CORR_SIGMA_MS}"
          --no-cache
          "${EDGES_ARGS[@]}"
          --out-dir "${eval_out}"
        )
        if [[ "$need_eval" -eq 1 ]]; then
          echo "[eval]  ${eval_cmd[*]}"
          if [[ "$DRY_RUN" -eq 0 ]]; then
            "${eval_cmd[@]}"
          fi
        else
          echo "[skip] eval outputs exist: ${eval_summary_json}"
        fi
      done
    done
  done
done

# Baseline: evaluate the stock/published model once at the end (ENS2-only).
baseline_out="${EVAL_ROOT}/baseline_ens2_published"
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
echo "[baseline] ${baseline_cmd[*]}"
if [[ -f "$baseline_summary_json" && -f "$baseline_summary_csv" && "$FORCE_EVAL" -eq 0 ]]; then
  echo "[skip] baseline eval outputs exist: ${baseline_summary_json}"
else
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${baseline_cmd[@]}"
  fi
fi
