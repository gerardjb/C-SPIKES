#!/usr/bin/env python3
"""
Convenience launcher for the new pipeline CLI without requiring an installed package.

Usage mirrors `python -m c_spikes.cli.run` but adds the repo's src/ to PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from c_spikes.cli.run import main  # noqa: E402

if __name__ == "__main__":
    main()
