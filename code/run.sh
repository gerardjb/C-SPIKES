#!/usr/bin/env bash
set -eo pipefail

usage() {
    cat <<'EOF'
C-SPIKES Code Ocean capsule entry point.

Usage:
  ./run.sh [--dataset-percent PERCENT] [stage ...]

Stages:
  setup        Write a run manifest and create standard output directories.
  quickcheck   Run lightweight import checks for required Python backends.
  smoke        Check GPU/backend visibility and run one epoch of inference.
  inference    Run the reviewer-facing manuscript inference workflow.
  biophys-ml   Run the BiophysML regeneration/checksum demo.
  all          Expand to: setup quickcheck smoke inference

Defaults:
  If no stage is supplied, C_SPIKES_RUN_STAGES is used.
  If C_SPIKES_RUN_STAGES is unset, the default is: setup quickcheck inference.

Common environment overrides:
  C_SPIKES_DATA_DIR       Data asset root. Default: ../data
  C_SPIKES_RESULTS_DIR    Persisted output root. Default: ../results
  C_SPIKES_SCRATCH_DIR    Temporary working root. Default: ../scratch
  C_SPIKES_DATASET_PERCENT
                          Optional inference expansion percent, >0 and <=100.
                          Unset uses curated default epochs.
  C_SPIKES_INFERENCE_PLAN_ONLY
                          Set to 1 to write the inference plan without running it.
  C_SPIKES_JG8F_BM_SIGMA  PGAS bm_sigma for jGCaMP8f inference. Default: 0.03
  C_SPIKES_JG8M_BM_SIGMA  PGAS bm_sigma for jGCaMP8m inference. Default: 0.05
  C_SPIKES_QUICKCHECK     Set to 0 to skip quickcheck inside all/default workflows.
  C_SPIKES_USE_SOURCE_TREE
                          Set to 1 to import c_spikes from code/src instead of
                          the installed package. Default: 0.
  C_SPIKES_SMOKE_REQUIRE_GPU
                           Set to 0 to allow smoke tests without a visible GPU.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
capsule_root="$(cd "${script_dir}/.." && pwd)"

export C_SPIKES_CODE_DIR="${C_SPIKES_CODE_DIR:-${script_dir}}"
export C_SPIKES_DATA_DIR="${C_SPIKES_DATA_DIR:-${capsule_root}/data}"
export C_SPIKES_RESULTS_DIR="${C_SPIKES_RESULTS_DIR:-${capsule_root}/results}"
export C_SPIKES_SCRATCH_DIR="${C_SPIKES_SCRATCH_DIR:-${capsule_root}/scratch}"

try_activate_conda() {
    if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
        return 0
    fi
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    fi
    if command -v module >/dev/null 2>&1; then
        module load anaconda3/2024.6 >/dev/null 2>&1 || true
    fi
    if ! command -v conda >/dev/null 2>&1; then
        return 0
    fi

    local conda_base target_env candidate
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "${conda_base}/etc/profile.d/conda.sh"
    fi

    target_env="${C_SPIKES_CONDA_ENV:-}"
    if [[ -z "${target_env}" ]]; then
        for candidate in c_spikes_co c_spikes; do
            if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "${candidate}"; then
                target_env="${candidate}"
                break
            fi
        done
    fi
    if [[ -n "${target_env}" ]]; then
        conda activate "${target_env}" || true
    fi
}

try_activate_conda

if [[ -f /etc/profile.d/c_spikes_build.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/c_spikes_build.sh
fi

mkdir -p \
    "${C_SPIKES_RESULTS_DIR}" \
    "${C_SPIKES_RESULTS_DIR}/logs" \
    "${C_SPIKES_SCRATCH_DIR}" \
    "${C_SPIKES_SCRATCH_DIR}/mpl_cache" \
    "${C_SPIKES_SCRATCH_DIR}/python_cache"

if [[ "${C_SPIKES_USE_SOURCE_TREE:-0}" == "1" ]]; then
    export PYTHONPATH="${C_SPIKES_CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${C_SPIKES_SCRATCH_DIR}/python_cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${C_SPIKES_SCRATCH_DIR}/mpl_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${C_SPIKES_SCRATCH_DIR}/mpl_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export C_SPIKES_TF_SUPPRESS_RUNTIME_STDERR="${C_SPIKES_TF_SUPPRESS_RUNTIME_STDERR:-1}"

prepend_ld_path() {
    local path="$1"
    if [[ -d "${path}" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${path}:"*) ;;
            *) export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
        esac
    fi
}

prepend_nvidia_python_libs() {
    local libs
    libs="$(python - <<'PY'
import glob
import site

roots = []
for root in site.getsitepackages():
    roots.extend(glob.glob(root + "/nvidia/*/lib"))
print(":".join(sorted(roots)))
PY
)"
    if [[ -n "${libs}" ]]; then
        export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

if command -v python >/dev/null 2>&1; then
    prepend_nvidia_python_libs
fi
prepend_ld_path "${VCPKG_ROOT:-/opt/vcpkg}/installed/${VCPKG_DEFAULT_TRIPLET:-x64-linux}/lib"
prepend_ld_path "${CUDA_HOME:-/usr/local/cuda}/lib64"

cd "${C_SPIKES_CODE_DIR}"

ensure_pgas_backend() {
    if [[ "${C_SPIKES_SKIP_PGAS_BUILD:-0}" == "1" ]]; then
        return 0
    fi

    if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("c_spikes.pgas.pgas_bound")
PY
    then
        return 0
    fi

    if [[ ! -f "${C_SPIKES_CODE_DIR}/pyproject.toml" ]]; then
        echo "[run.sh] PGAS backend is not importable and ${C_SPIKES_CODE_DIR}/pyproject.toml is missing." >&2
        return 1
    fi

    echo "[run.sh] PGAS backend is not importable; building C-SPIKES package."
    python -m pip install --no-deps --no-build-isolation -v "${C_SPIKES_CODE_DIR}"
    python - <<'PY'
import importlib
module = importlib.import_module("c_spikes.pgas.pgas_bound")
print(f"[run.sh] PGAS backend import ok: {getattr(module, '__backend__', 'unknown')}")
PY
}

stage_setup() {
    echo "[run.sh] setup"
    ensure_pgas_backend
    python - <<'PY'
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

results_dir = Path(os.environ["C_SPIKES_RESULTS_DIR"])
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "code_dir": os.environ["C_SPIKES_CODE_DIR"],
    "data_dir": os.environ["C_SPIKES_DATA_DIR"],
    "results_dir": os.environ["C_SPIKES_RESULTS_DIR"],
    "scratch_dir": os.environ["C_SPIKES_SCRATCH_DIR"],
    "packages": {},
}
for name in ("c_spikes", "tensorflow", "torch", "numpy", "scipy", "matplotlib"):
    try:
        manifest["packages"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        manifest["packages"][name] = None

out = results_dir / "run_manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[run.sh] wrote {out}")
PY
}

stage_quickcheck() {
    if [[ "${C_SPIKES_QUICKCHECK:-1}" != "1" ]]; then
        echo "[run.sh] quickcheck skipped (C_SPIKES_QUICKCHECK=${C_SPIKES_QUICKCHECK})"
        return 0
    fi
    echo "[run.sh] quickcheck"
    ensure_pgas_backend
    python -m pytest -q tests/test_dependency_imports.py
}

stage_smoke() {
    ensure_pgas_backend
    run_python_stage \
        "smoke" \
        "scripts/code_ocean_smoke_test.py" \
        "${C_SPIKES_RESULTS_DIR}/smoke"
}

run_python_stage() {
    local stage="$1"
    local script="$2"
    local out_dir="$3"
    if [[ ! -f "${script}" ]]; then
        echo "[run.sh] stage '${stage}' is not implemented yet." >&2
        echo "[run.sh] expected script: ${script}" >&2
        return 2
    fi
    mkdir -p "${out_dir}"
    echo "[run.sh] ${stage}"
    python "${script}" \
        --data-dir "${C_SPIKES_DATA_DIR}" \
        --results-dir "${out_dir}" \
        --scratch-dir "${C_SPIKES_SCRATCH_DIR}"
}

stage_inference() {
    ensure_pgas_backend
    run_python_stage \
        "inference" \
        "scripts/code_ocean_inference_demo.py" \
        "${C_SPIKES_RESULTS_DIR}/inference_parity"
}

stage_biophys_ml() {
    ensure_pgas_backend
    run_python_stage \
        "biophys-ml" \
        "scripts/code_ocean_biophys_ml_demo.py" \
        "${C_SPIKES_RESULTS_DIR}/biophys_ml_parity"
}

normalize_stage() {
    echo "$1" | tr '_' '-'
}

run_stage() {
    local stage
    stage="$(normalize_stage "$1")"
    case "${stage}" in
        setup) stage_setup ;;
        quickcheck) stage_quickcheck ;;
        smoke) stage_smoke ;;
        inference) stage_inference ;;
        biophys-ml) stage_biophys_ml ;;
        all)
            stage_setup
            stage_quickcheck
            stage_smoke
            stage_inference
            stage_biophys_ml
            ;;
        help|-h|--help) usage ;;
        *)
            echo "[run.sh] unknown stage: ${stage}" >&2
            usage >&2
            return 2
            ;;
    esac
}

stages=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dataset-percent)
            if [[ "$#" -lt 2 ]]; then
                echo "[run.sh] --dataset-percent requires a value." >&2
                exit 2
            fi
            export C_SPIKES_DATASET_PERCENT="$2"
            shift 2
            ;;
        --dataset-percent=*)
            export C_SPIKES_DATASET_PERCENT="${1#*=}"
            shift
            ;;
        help|-h|--help)
            usage
            exit 0
            ;;
        *)
            stages+=("$1")
            shift
            ;;
    esac
done


if [[ "${#stages[@]}" -eq 0 ]]; then
    read -r -a stages <<< "${C_SPIKES_RUN_STAGES:-setup quickcheck inference}"
fi

for stage in "${stages[@]}"; do
    run_stage "${stage}"
done
