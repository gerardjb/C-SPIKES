"""Profile selection occurs before imports, so native-core tests need no ML/GUI stack."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = json.loads((Path(__file__).parent / "profiles.json").read_text())


def pytest_addoption(parser):
    parser.addoption("--csp-profile", choices=tuple(PROFILES) + ("all",), default="native_core",
                     help="Explicit supported-contract test profile; see tests/README.md")


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    if path.is_dir() and path.name == "native": return True
    if path.suffix != ".py" or not path.name.startswith("test_"): return None
    profile = config.getoption("--csp-profile")
    if profile == "all": return None
    return path.relative_to(ROOT).as_posix() not in PROFILES[profile]


def pytest_collection_finish(session):
    if not session.items:
        raise pytest.UsageError("Selected profile collected no tests after selection")
