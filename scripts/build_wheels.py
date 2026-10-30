"""Run cibuildwheel for one explicit CPU/CUDA artifact set (no publishing)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys

from build_native import ARCHITECTURES, msvc_environment

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux", "windows"), default=platform.system().lower())
    parser.add_argument("--print-build-identifiers", action="store_true")
    parser.add_argument("--cuda-arch", default="TURING75", choices=ARCHITECTURES)
    args = parser.parse_args()
    system = args.platform
    if system not in {"linux", "windows"}:
        parser.error("Only Linux and Windows wheel builds are configured")
    gpu = args.backend == "cuda"
    settings = {
        "cmake.define.CMAKE_TOOLCHAIN_FILE": "{project}/build/tools/vcpkg/scripts/buildsystems/vcpkg.cmake",
        "cmake.define.VCPKG_INSTALLED_DIR": "{project}/build/vcpkg-installed",
        "cmake.define.VCPKG_TARGET_TRIPLET": f"x64-{system}-csp",
        "cmake.define.VCPKG_MANIFEST_MODE": "OFF",
        "cmake.define.PGAS_BUILD_GPU": "ON" if gpu else "OFF",
        "cmake.define.Kokkos_ENABLE_CUDA": "ON" if gpu else "OFF",
    }
    if gpu:
        settings[f"cmake.define.Kokkos_ARCH_{args.cuda_arch}"] = "ON"
        settings["cmake.define.CMAKE_CUDA_ARCHITECTURES"] = ARCHITECTURES[args.cuda_arch]
        settings["cmake.define.CMAKE_CUDA_COMPILER"] = "{project}/build/tools/cuda/bin/nvcc" + (".exe" if system == "windows" else "")
    env = os.environ.copy()
    if system == "windows" and platform.system() == "Windows" and not args.print_build_identifiers:
        env = msvc_environment(env)
    env["CIBW_CONFIG_SETTINGS"] = " ".join(f'{key}="{value}"' for key, value in settings.items())
    env["CIBW_ARCHS"] = "AMD64" if system == "windows" else "x86_64"
    env["CIBW_ENVIRONMENT"] = f"CSP_WHEEL_BACKEND={args.backend} CMAKE_GENERATOR=Ninja CMAKE_BUILD_PARALLEL_LEVEL=2 VCPKG_MAX_CONCURRENCY=2 OMP_NUM_THREADS=1 OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1"
    if system == "linux":
        toolchain = "/opt/rh/gcc-toolset-12/root/usr/bin"
        env["CIBW_ENVIRONMENT"] += f' CC={toolchain}/gcc CXX={toolchain}/g++ FC={toolchain}/gfortran PATH="{toolchain}:$PATH"'
    if gpu:
        # Standard hosted runners have no CUDA device. Runtime qualification is
        # performed with build_native.py --test on an actual GPU, never faked.
        env["CIBW_TEST_SKIP"] = "*"
        if system == "linux":
            env["CIBW_REPAIR_WHEEL_COMMAND"] = 'LD_LIBRARY_PATH="{project}/build/tools/cuda/lib" auditwheel repair --exclude libcuda.so.1 -w {dest_dir} {wheel}'
        else:
            env["CIBW_REPAIR_WHEEL_COMMAND"] = 'delvewheel repair --exclude nvcuda.dll --add-path "{project}/build/vcpkg-installed/x64-windows-csp/bin;{project}/build/tools/cuda/bin" -w {dest_dir} {wheel}'
    command = [sys.executable, "-m", "cibuildwheel", "--platform", system,
               "--output-dir", args.output_dir.resolve(), ROOT]
    if args.print_build_identifiers:
        command.append("--print-build-identifiers")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
