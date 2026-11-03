"""Reject unsupported hosts rather than silently using an x86 dependency triplet."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_native


@pytest.mark.parametrize("system,machine,expected", [
    ("Linux", "x86_64", "linux"), ("Windows", "AMD64", "windows"),
    ("Darwin", "arm64", "macos"), ("Darwin", "aarch64", "macos"),
])
def test_supported_build_hosts(monkeypatch, system, machine, expected):
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(build_native.platform, "machine", lambda: machine)
    assert build_native.host_platform() == expected


@pytest.mark.parametrize("system,machine", [
    ("Darwin", "x86_64"), ("Linux", "aarch64"), ("Windows", "ARM64"), ("Linux", "i686"),
])
def test_unsupported_build_hosts(monkeypatch, system, machine):
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(build_native.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Supported build hosts"):
        build_native.host_platform()
