"""Reject incomplete, mislabeled and unsuitable artifacts before PyPI upload."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_release import validate_bundle, version_for_tag


def metadata(version):
    return f"Metadata-Version: 2.1\nName: c-spikes\nVersion: {version}\nRequires-Python: >=3.11\n\n".encode()


def bundle(directory, version="0.2.0", *, backend="combined", extra_member=None):
    for python in ("cp311", "cp312", "cp313", "cp314"):
        for platform in ("manylinux_2_27_x86_64.manylinux_2_28_x86_64", "win_amd64"):
            path = directory / f"c_spikes-{version}-{python}-{python}-{platform}.whl"
            with zipfile.ZipFile(path, "w") as wheel:
                wheel.writestr(f"c_spikes-{version}.dist-info/METADATA", metadata(version))
                wheel.writestr("c_spikes/pgas/backends.py", "")
                suffix = "pyd" if platform == "win_amd64" else "so"
                for name in (("cpu", "gpu") if backend == "combined" else (backend,)):
                    wheel.writestr(f"c_spikes/pgas/pgas_bound_{name}.{python}.{suffix}", b"test fixture")
                if extra_member:
                    wheel.writestr(extra_member, b"test fixture")
    with tarfile.open(directory / f"c_spikes-{version}.tar.gz", "w:gz") as sdist:
        info = tarfile.TarInfo(f"c_spikes-{version}/PKG-INFO")
        data = metadata(version)
        info.size = len(data)
        sdist.addfile(info, io.BytesIO(data))
    return directory


@pytest.mark.parametrize("version", ["0.2.0", "0.2.0rc1", "0.2.0.post1"])
def test_complete_release_hashes_every_file(tmp_path, version):
    files = validate_bundle(bundle(tmp_path, version), version_for_tag("v" + version))
    assert len(files) == 9
    assert files == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("tag", ["main", "0.2.0", "v01.2.0", "v0.2.0rc01", "v0.2.0+cu129", "v0.2.0.dev1", "v0.2.0\n"])
def test_invalid_tags(tag):
    with pytest.raises(ValueError):
        version_for_tag(tag)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_both_extensions_required(tmp_path, backend):
    with pytest.raises(ValueError, match="extension"):
        validate_bundle(bundle(tmp_path, backend=backend), "0.2.0")


@pytest.mark.parametrize("mutation", ["missing-wheel", "missing-sdist", "extra", "duplicate-slot", "local-cuda-version", "wrong-abi", "local-linux", "high-glibc"])
def test_bad_artifact_sets(tmp_path, mutation):
    bundle(tmp_path)
    wheel = next(tmp_path.glob("*manylinux*.whl"))
    if mutation == "missing-wheel":
        wheel.unlink()
    elif mutation == "missing-sdist":
        next(tmp_path.glob("*.tar.gz")).unlink()
    elif mutation == "extra":
        (tmp_path / "unexpected.txt").write_text("not a distribution")
    elif mutation == "duplicate-slot":
        other = next(tmp_path.glob("*cp312*manylinux*.whl"))
        other.rename(other.with_name(other.name.replace("cp312", "cp311").replace("manylinux_2_27_x86_64.", "")))
    elif mutation == "local-cuda-version":
        wheel.rename(wheel.with_name(wheel.name.replace("0.2.0-", "0.2.0+cu129-")))
    elif mutation == "wrong-abi":
        wheel.rename(wheel.with_name(wheel.name.replace("-cp311-cp311-", "-cp311-abi3-")))
    else:
        replacement = "linux_x86_64" if mutation == "local-linux" else "manylinux_2_39_x86_64"
        wheel.rename(wheel.with_name(wheel.name.replace("manylinux_2_27_x86_64.manylinux_2_28_x86_64", replacement)))
    with pytest.raises(ValueError):
        validate_bundle(tmp_path, "0.2.0")


@pytest.mark.parametrize("member", ["c_spikes.libs/libcuda.so.1", "c_spikes.libs/libcuda-abcd.so.1", "c_spikes.libs/nvcuda.dll"])
def test_no_bundled_nvidia_driver(tmp_path, member):
    with pytest.raises(ValueError, match="NVIDIA driver"):
        validate_bundle(bundle(tmp_path, extra_member=member), "0.2.0")


@pytest.mark.parametrize("field", ["name", "version", "python", "loader"])
def test_wheel_contents_must_match_release(tmp_path, field):
    bundle(tmp_path)
    path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(path) as wheel:
        contents = {n: wheel.read(n) for n in wheel.namelist()}
    meta = "c_spikes-0.2.0.dist-info/METADATA"
    if field == "loader":
        del contents["c_spikes/pgas/backends.py"]
    else:
        old, new = {"name": (b"Name: c-spikes", b"Name: unrelated"),
                    "version": (b"Version: 0.2.0", b"Version: 0.1.0"),
                    "python": (b">=3.11", b">=3.10")}[field]
        contents[meta] = contents[meta].replace(old, new)
    with zipfile.ZipFile(path, "w") as wheel:
        for name, data in contents.items():
            wheel.writestr(name, data)
    with pytest.raises(ValueError):
        validate_bundle(tmp_path, "0.2.0")


def test_sdist_metadata_must_match_tag(tmp_path):
    bundle(tmp_path)
    with tarfile.open(tmp_path / "c_spikes-0.2.0.tar.gz", "w:gz") as sdist:
        info = tarfile.TarInfo("c_spikes-0.2.0/PKG-INFO")
        data = metadata("0.1.0")
        info.size = len(data)
        sdist.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="version"):
        validate_bundle(tmp_path, "0.2.0")
