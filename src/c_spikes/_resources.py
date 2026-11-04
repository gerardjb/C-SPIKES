"""Paths to bundled PGAS inputs, independent of the working directory.

Native wheels are installed unpacked, so these paths can be passed directly to
the C++ file readers. The source checkout keeps its canonical constants in
parameter_files; CMake copies them into the wheel at installation time.
"""
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
PGAS_GPARAM_DIR = _PACKAGE_ROOT / "pgas"
PGAS_CONSTANTS_DIR = PGAS_GPARAM_DIR / "constants"

# Editable installs load Python from src/ and retain the checkout's inputs.
_SOURCE_ROOT = _PACKAGE_ROOT.parent.parent
if not PGAS_CONSTANTS_DIR.is_dir() and (_SOURCE_ROOT / "pyproject.toml").is_file():
    PGAS_CONSTANTS_DIR = _SOURCE_ROOT / "parameter_files"

DEFAULT_PGAS_CONSTANTS = PGAS_CONSTANTS_DIR / "constants_GCaMP8_soma.json"
DEFAULT_GPARAM = PGAS_GPARAM_DIR / "20230525_gold.dat"
