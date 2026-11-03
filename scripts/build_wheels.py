"""Build combined CPU/CUDA distribution wheels; CPU-only builds are opt-in."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys

from build_native import ARCHITECTURES, CUDA_RELEASE, cuda_settings, msvc_environment, wheel_version

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux", "windows"), default=platform.system().lower())
    parser.add_argument("--print-build-identifiers", action="store_true")
    parser.add_argument("--cuda-arch", default="PORTABLE", choices=["PORTABLE", *ARCHITECTURES])
    parser.add_argument("--version", help="Public version for a release build (otherwise derive from Git)")
    args = parser.parse_args()
    system = args.platform
    if system not in {"linux", "windows"}:
        parser.error("Only Linux and Windows wheel builds are configured")
    gpu = args.backend == "cuda"
    cuda_root = f"{{project}}/build/tools/cuda-{CUDA_RELEASE}"
    settings = {
        "cmake.define.CMAKE_TOOLCHAIN_FILE": "{project}/build/tools/vcpkg/scripts/buildsystems/vcpkg.cmake",
        "cmake.define.VCPKG_INSTALLED_DIR": "{project}/build/vcpkg-installed",
        "cmake.define.VCPKG_TARGET_TRIPLET": f"x64-{system}-csp",
        "cmake.define.VCPKG_MANIFEST_MODE": "OFF",
        "cmake.define.PGAS_BUILD_GPU": "ON" if gpu else "OFF",
        "cmake.define.Kokkos_ENABLE_CUDA": "ON" if gpu else "OFF",
        "cmake.define.CSP_COMBINED_WHEEL": "ON" if gpu else "OFF",
    }
    if gpu:
        settings.update(cuda_settings(args.cuda_arch))
        settings["cmake.define.CMAKE_CUDA_COMPILER"] = cuda_root + "/bin/nvcc" + (".exe" if system == "windows" else "")
    env = os.environ.copy()
    if system == "windows" and platform.system() == "Windows" and not args.print_build_identifiers:
        env = msvc_environment(env)
    env["CIBW_CONFIG_SETTINGS"] = " ".join(f'{key}="{value}"' for key, value in settings.items())
    env["CIBW_ARCHS"] = "AMD64" if system == "windows" else "x86_64"
    env["CIBW_ENVIRONMENT"] = f"CSP_WHEEL_BACKEND={args.backend} CMAKE_GENERATOR=Ninja CMAKE_BUILD_PARALLEL_LEVEL=2 VCPKG_MAX_CONCURRENCY=2 OMP_NUM_THREADS=1 OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1"
    if gpu or args.version:
        version = wheel_version(args.version)
        env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_C_SPIKES"] = version
        env["CIBW_ENVIRONMENT"] += f" SETUPTOOLS_SCM_PRETEND_VERSION_FOR_C_SPIKES={version}"
    if system == "linux":
        toolchain = "/opt/rh/gcc-toolset-12/root/usr/bin"
        env["CIBW_ENVIRONMENT"] += f' CC={toolchain}/gcc CXX={toolchain}/g++ FC={toolchain}/gfortran PATH="{toolchain}:$PATH"'
    if gpu:
        # CPU must work without a driver on hosted runners. GPU execution is
        # qualified separately with build_native.py --backend cuda --test.
        env["CIBW_TEST_COMMAND"] = " && ".join([
            'python "{project}/tests/packaging/wheel_smoke.py" --backend cpu --import-order numpy-first',
            'python "{project}/tests/packaging/wheel_smoke.py" --backend cpu --import-order native-first',
            'python "{project}/tests/packaging/combined_wheel_smoke.py" --no-gpu',
            'python -m pytest "{project}/tests" --csp-profile=native_core',
            'python -m pytest "{project}/tests" --csp-profile=python_contracts',
        ])
        if system == "linux":
            env["CIBW_REPAIR_WHEEL_COMMAND"] = (
                f'LD_LIBRARY_PATH="{cuda_root}/lib" auditwheel repair --exclude libcuda.so.1 '
                '-w "{dest_dir}" "{wheel}"')
        else:
            env["CIBW_REPAIR_WHEEL_COMMAND"] = (
                'python "{project}/scripts/repair_windows_wheel.py" repair --exclude nvcuda.dll '
                f'--add-path "{{project}}/build/vcpkg-installed/x64-windows-csp/bin;{cuda_root}/bin" '
                '-w "{dest_dir}" "{wheel}"')
        # Inspect the repaired wheel even on hosted runners without a GPU.
        env["CIBW_REPAIR_WHEEL_COMMAND"] += (
            ' && python "{project}/scripts/check_cuda_wheel.py" --wheel-dir "{dest_dir}" '
            f'--cuda-root "{cuda_root}" --cuda-arch {args.cuda_arch}')
    command = [sys.executable, "-m", "cibuildwheel", "--platform", system,
               "--output-dir", args.output_dir.resolve(), ROOT]
    if args.print_build_identifiers:
        command.append("--print-build-identifiers")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
