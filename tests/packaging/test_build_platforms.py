"""Reject unsupported hosts rather than silently using an x86 dependency triplet."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_native
import build_wheels
import prepare_cuda


@pytest.mark.parametrize("system,machine,expected,triplet", [
    ("Linux", "x86_64", "linux", "x64-linux-csp"),
    ("Linux", "aarch64", "linux", "arm64-linux-csp"),
    ("Linux", "arm64", "linux", "arm64-linux-csp"),
    ("Windows", "AMD64", "windows", "x64-windows-csp"),
    ("Darwin", "arm64", "macos", "arm64-osx-csp"),
    ("Darwin", "aarch64", "macos", "arm64-osx-csp"),
])
def test_supported_build_hosts(monkeypatch, system, machine, expected, triplet):
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(build_native.platform, "machine", lambda: machine)
    assert build_native.host_platform() == expected
    assert build_native.TRIPLETS[expected, build_native.host_architecture()] == triplet


@pytest.mark.parametrize("system,machine", [
    ("Darwin", "x86_64"), ("Windows", "ARM64"), ("Linux", "i686"), ("Linux", "armv7l"),
])
def test_unsupported_build_hosts(monkeypatch, system, machine):
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(build_native.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Supported build hosts"):
        build_native.host_platform()


@pytest.mark.parametrize("machine,arch,triplet,backend", [
    ("aarch64", "aarch64", "arm64-linux-csp", "cuda"),
    ("x86_64", "x86_64", "x64-linux-csp", "cuda"),
])
def test_linux_wheel_defaults_match_host(monkeypatch, tmp_path, machine, arch, triplet, backend):
    monkeypatch.setattr(build_native.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_native.platform, "machine", lambda: machine)
    monkeypatch.setattr(sys, "argv", ["build_wheels.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(build_wheels, "wheel_version", lambda _: "0.2.0")
    calls = []
    monkeypatch.setattr(build_wheels.subprocess, "run", lambda command, **kwargs: calls.append(kwargs["env"]))
    build_wheels.main()
    assert len(calls) == 1
    env = calls[0]
    assert env["CIBW_ARCHS"] == arch
    assert f'CMAKE_TOOLCHAIN_FILE="{{project}}/build/tools/vcpkg/scripts/buildsystems/vcpkg.cmake"' in env["CIBW_CONFIG_SETTINGS"]
    assert f'VCPKG_TARGET_TRIPLET="{triplet}"' in env["CIBW_CONFIG_SETTINGS"]
    assert f"CSP_WHEEL_BACKEND={backend}" in env["CIBW_ENVIRONMENT"]
    assert f'PGAS_BUILD_GPU="{"ON" if backend == "cuda" else "OFF"}"' in env["CIBW_CONFIG_SETTINGS"]


@pytest.mark.parametrize("script,options", [
    (build_wheels, ["--output-dir", "unused", "--backend", "cuda"]),
    (build_native, ["--backend", "cuda", "--cuda-arch", "PORTABLE"]),
])
def test_macos_cuda_rejected_before_provisioning(monkeypatch, capsys, script, options):
    monkeypatch.setattr(build_native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build_native.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(sys, "argv", [script.__file__, *options])
    with pytest.raises(SystemExit):
        script.main()
    assert "use --backend cpu on macOS" in capsys.readouterr().err


def test_cross_arch_build_rejected(monkeypatch, capsys):
    monkeypatch.setattr(build_native.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_native.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "argv", ["build_wheels.py", "--output-dir", "unused", "--arch", "arm64"])
    with pytest.raises(SystemExit):
        build_wheels.main()
    assert "native runner" in capsys.readouterr().err


@pytest.mark.parametrize("system,machine,target", [
    ("Linux", "aarch64", "linux-sbsa"), ("Linux", "arm64", "linux-sbsa"),
    ("Linux", "x86_64", "linux-x86_64"), ("Windows", "AMD64", "windows-x86_64"),
])
def test_cuda_toolkit_matches_host(monkeypatch, system, machine, target):
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(build_native.platform, "machine", lambda: machine)
    assert prepare_cuda.native_cuda_platform() == target


@pytest.mark.parametrize("system,machine", [("Darwin", "arm64"), ("Windows", "ARM64"), ("Linux", "armv7l")])
def test_unsupported_cuda_host_cannot_download_x86_tools(monkeypatch, system, machine):
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    monkeypatch.setattr(build_native.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Pinned CUDA tools support"):
        prepare_cuda.native_cuda_platform()
