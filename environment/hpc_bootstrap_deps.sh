#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

usage() {
    cat <<'EOF'
Stage native build dependencies for environment/hpc_build.sh.

This mirrors the native libraries installed by environment/Dockerfile without
requiring sudo on an HPC system. Docker gets these from apt:
  libarmadillo-dev libboost-dev libgsl-dev libjsoncpp-dev libopenblas-dev

The HPC bootstrap stages their vcpkg equivalents:
  openblas[dynamic-arch] armadillo boost-circular-buffer gsl jsoncpp

It also clones Kokkos at the same version used by the Dockerfile.

Useful overrides:
  C_SPIKES_DEPS_ROOT    Dependency cache root (default: $SCRATCH/c_spikes_deps)
  VCPKG_ROOT            vcpkg checkout path
  VCPKG_REF             vcpkg tag/branch (default: 2026.03.18)
  VCPKG_DEFAULT_TRIPLET vcpkg triplet (default: x64-linux)
  KOKKOS_SOURCE_DIR     Kokkos source checkout path
  KOKKOS_REF            Kokkos tag/branch (default: 4.3.01)
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

: "${C_SPIKES_DEPS_ROOT:=${SCRATCH:-/scratch/gpfs/WANG/${USER}}/c_spikes_deps}"
: "${VCPKG_ROOT:=${C_SPIKES_DEPS_ROOT}/vcpkg}"
: "${VCPKG_DEFAULT_TRIPLET:=x64-linux}"
: "${VCPKG_REF:=2026.03.18}"
: "${KOKKOS_SOURCE_DIR:=${C_SPIKES_DEPS_ROOT}/kokkos-src}"
: "${KOKKOS_REF:=4.3.01}"

export VCPKG_DISABLE_METRICS=1

mkdir -p "${C_SPIKES_DEPS_ROOT}"

if [[ ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
    rm -rf "${VCPKG_ROOT}"
    git clone --depth 1 --branch "${VCPKG_REF}" https://github.com/microsoft/vcpkg "${VCPKG_ROOT}"
    "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
fi

"${VCPKG_ROOT}/vcpkg" install \
    --triplet "${VCPKG_DEFAULT_TRIPLET}" \
    --recurse \
    --clean-after-build \
    "openblas[dynamic-arch]" \
    armadillo \
    boost-circular-buffer \
    gsl \
    jsoncpp

if [[ ! -f "${KOKKOS_SOURCE_DIR}/CMakeLists.txt" ]]; then
    rm -rf "${KOKKOS_SOURCE_DIR}"
    git clone --depth 1 --branch "${KOKKOS_REF}" https://github.com/kokkos/kokkos.git "${KOKKOS_SOURCE_DIR}"
fi

cat <<EOF
HPC C-SPIKES dependencies are ready.
  repo: ${repo_root}
  VCPKG_ROOT=${VCPKG_ROOT}
  KOKKOS_SOURCE_DIR=${KOKKOS_SOURCE_DIR}
EOF
