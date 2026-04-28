#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
code_dir="${repo_root}/code"
requirements_file="${script_dir}/requirements.txt"

if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
    module load anaconda3/2024.6 >/dev/null 2>&1 || true
    if [[ -n "${C_SPIKES_CUDA_MODULE:-}" ]]; then
        module load "${C_SPIKES_CUDA_MODULE}"
    else
        module load cudatoolkit/12.9 >/dev/null 2>&1 \
            || module load cudatoolkit/12.8 >/dev/null 2>&1 \
            || module load cudatoolkit/12.6 >/dev/null 2>&1 \
            || true
    fi
fi

if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${C_SPIKES_CONDA_ENV:-c_spikes_co}"
fi

detect_cuda_arch() {
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    cap="${cap/./}"
    case "${cap}" in
        75) echo "TURING75 75" ;;
        80) echo "AMPERE80 80" ;;
        86) echo "AMPERE86 86" ;;
        89) echo "ADA89 89" ;;
        90) echo "HOPPER90 90" ;;
        *)
            echo "Could not map GPU compute capability '${cap}'. Set C_SPIKES_KOKKOS_ARCH and C_SPIKES_CUDA_ARCH." >&2
            return 1
            ;;
    esac
}

if [[ -z "${C_SPIKES_KOKKOS_ARCH:-}" || -z "${C_SPIKES_CUDA_ARCH:-}" ]]; then
    read -r detected_kokkos_arch detected_cuda_arch < <(detect_cuda_arch)
    : "${C_SPIKES_KOKKOS_ARCH:=${detected_kokkos_arch}}"
    : "${C_SPIKES_CUDA_ARCH:=${detected_cuda_arch}}"
fi

: "${C_SPIKES_DEPS_ROOT:=${SCRATCH:-/scratch/gpfs/WANG/${USER}}/c_spikes_deps}"
: "${VCPKG_ROOT:=${C_SPIKES_DEPS_ROOT}/vcpkg}"
: "${VCPKG_DEFAULT_TRIPLET:=x64-linux}"
: "${KOKKOS_SOURCE_DIR:=${C_SPIKES_DEPS_ROOT}/kokkos-src}"

if [[ ! -x "${VCPKG_ROOT}/vcpkg" || ! -f "${KOKKOS_SOURCE_DIR}/CMakeLists.txt" ]]; then
    "${script_dir}/hpc_bootstrap_deps.sh"
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    else
        CUDA_HOME="/usr/local/cuda"
    fi
fi

nvidia_python_libs=""
if command -v python >/dev/null 2>&1; then
    nvidia_python_libs="$(python - <<'PY'
import glob
import site

print(":".join(sorted(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib"))))
PY
)"
fi

export VCPKG_ROOT
export VCPKG_DEFAULT_TRIPLET
export KOKKOS_SOURCE_DIR
export FETCHCONTENT_SOURCE_DIR_KOKKOS="${KOKKOS_SOURCE_DIR}"
export PGAS_KOKKOS_ARCH="${C_SPIKES_KOKKOS_ARCH}"
export PGAS_CUDA_ARCHITECTURES="${C_SPIKES_CUDA_ARCH}"
export CUDA_HOME
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CXX="${KOKKOS_SOURCE_DIR}/bin/nvcc_wrapper"
export NVCC_WRAPPER_DEFAULT_COMPILER="${NVCC_WRAPPER_DEFAULT_COMPILER:-$(command -v g++)}"
export CMAKE_PREFIX_PATH="${VCPKG_ROOT}/installed/${VCPKG_DEFAULT_TRIPLET}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${VCPKG_ROOT}/installed/${VCPKG_DEFAULT_TRIPLET}/lib${nvidia_python_libs:+:${nvidia_python_libs}}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"
export SKBUILD_BUILD_VERBOSE="${SKBUILD_BUILD_VERBOSE:-true}"
export SKBUILD_BUILD_DIR="${SKBUILD_BUILD_DIR:-build/hpc-${C_SPIKES_CUDA_ARCH}}"

echo "Building C-SPIKES for ${PGAS_KOKKOS_ARCH} / CUDA arch ${PGAS_CUDA_ARCHITECTURES}"
if [[ ! -x "${CUDACXX}" ]]; then
    echo "CUDA compiler not found at ${CUDACXX}. Load a CUDA toolkit module or set CUDA_HOME." >&2
    exit 1
fi
if [[ ! -x "${CXX}" ]]; then
    echo "Kokkos nvcc_wrapper not found at ${CXX}. Run ${script_dir}/hpc_bootstrap_deps.sh first." >&2
    exit 1
fi

if [[ "${C_SPIKES_INSTALL_REQUIREMENTS:-0}" == "1" ]]; then
    python -m pip install -r "${requirements_file}"
fi

pip_install_args=(-e "${code_dir}" -v)
if [[ "${C_SPIKES_INSTALL_PROJECT_DEPS:-0}" != "1" ]]; then
    pip_install_args+=(--no-deps)
fi
python -m pip install "${pip_install_args[@]}"

python - <<'PY'
import importlib

for name in ("c_spikes.pgas.pgas_bound_cpu", "c_spikes.pgas.pgas_bound_gpu"):
    module = importlib.import_module(name)
    print(f"import ok: {module.__name__}")
PY

if [[ "${C_SPIKES_RUN_TESTS:-0}" == "1" ]]; then
    cd "${code_dir}"
    python -m pytest -q tests/test_dependency_imports.py
fi
