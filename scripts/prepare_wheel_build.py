"""cibuildwheel before-all: prepare pinned native dependencies and CUDA tools."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# CUDA 12.6 supports GCC <=13. Use one compiler for the C++, CUDA host and
# Fortran dependencies, including CPU wheels built in the same manylinux image.
if sys.platform == "linux":
    subprocess.run(["dnf", "install", "-y", "gcc-toolset-12-gcc", "gcc-toolset-12-gcc-c++",
                    "gcc-toolset-12-gcc-gfortran", "zip", "unzip", "perl"], check=True)
subprocess.run([sys.executable, ROOT / "scripts/build_native.py", "--dependencies-only"], check=True)
if os.environ.get("CSP_WHEEL_BACKEND") == "cuda":
    subprocess.run([sys.executable, ROOT / "scripts/prepare_cuda.py"], check=True)
