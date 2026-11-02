"""Release boundary tests: variants, completeness, immutability and index history."""
import copy
from email.message import EmailMessage
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_native import cuda_settings, wheel_version
from wheel_index import (GitHub, asset_url, build_index, make_manifest, public_version,
                         publish, release_manifests, validate_manifest)

REPO = "example/c-spikes"


def wheel_set(directory, version="0.2.0+cu129"):
    directory.mkdir(parents=True, exist_ok=True)
    for python in ("cp311", "cp312", "cp313", "cp314"):
        for platform in ("manylinux_2_27_x86_64.manylinux_2_28_x86_64", "win_amd64"):
            name = f"c_spikes-{version}-{python}-{python}-{platform}.whl"
            info = EmailMessage()
            info["Metadata-Version"] = "2.1"
            info["Name"] = "c-spikes"
            info["Version"] = version
            info["Requires-Python"] = ">=3.11"
            with zipfile.ZipFile(directory / name, "w") as package:
                package.writestr(f"c_spikes-{version}.dist-info/METADATA", info.as_bytes())
                suffix = "pyd" if platform == "win_amd64" else "so"
                for backend in ("cpu", "gpu"):
                    package.writestr(f"c_spikes/pgas/pgas_bound_{backend}.{python}.{suffix}", b"fixture")
    return directory


class FakeGitHub:
    repository = REPO
    api = "https://api.github.com/repos/" + REPO

    def __init__(self):
        self.assets = {}
        self.uploaded = []
        self.release = {"id": 1, "draft": False, "tag_name": "v0.2.0"}

    def request(self, url):
        return self.release

    def pages(self, path):
        if path == "releases":
            return [self.release]
        return [{"name": name, "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                 "size": len(data), "state": "uploaded"} for name, data in self.assets.items()]

    def download(self, tag, filename):
        return self.assets[filename]

    def upload(self, release, path):
        self.uploaded.append(path.name)
        self.assets[path.name] = path.read_bytes()


def test_complete_publish_is_idempotent_and_manifest_is_last(tmp_path):
    wheels = wheel_set(tmp_path)
    github = FakeGitHub()
    publish(github, wheels, "v0.2.0")
    assert len(github.uploaded) == 9
    assert github.uploaded[-1] == "c-spikes-cu129.json"
    assert len(list(release_manifests(github))) == 1
    publish(github, wheels, "v0.2.0")
    assert len(github.uploaded) == 9


def test_published_wheels_cannot_be_replaced(tmp_path):
    wheel_set(tmp_path)
    github = FakeGitHub()
    filename = next(tmp_path.glob("*.whl")).name
    github.assets[filename] = b"different existing bytes"
    with pytest.raises(ValueError, match="overwrite"):
        publish(github, tmp_path, "v0.2.0")
    assert not github.uploaded


def test_incomplete_matrix_never_uploads(tmp_path):
    wheel_set(tmp_path)
    next(tmp_path.glob("*win_amd64.whl")).unlink()
    github = FakeGitHub()
    with pytest.raises(ValueError, match="Incomplete"):
        publish(github, tmp_path, "v0.2.0")
    assert not github.uploaded


@pytest.mark.parametrize("version", ["0.2.0", "0.2.0+cu126", "0.3.0+cu129"])
def test_wrong_variant_or_release_rejected(tmp_path, version):
    wheel_set(tmp_path, version)
    with pytest.raises(ValueError, match="project/version"):
        make_manifest(tmp_path, REPO, "v0.2.0")


def test_filename_relabel_is_not_enough(tmp_path):
    wheel_set(tmp_path)
    path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(path) as package:
        contents = {name: package.read(name) for name in package.namelist()}
    metadata = next(name for name in contents if name.endswith("/METADATA"))
    contents[metadata] = contents[metadata].replace(b"Version: 0.2.0+cu129", b"Version: 0.2.0")
    with zipfile.ZipFile(path, "w") as package:
        for name, content in contents.items():
            package.writestr(name, content)
    with pytest.raises(ValueError, match="metadata disagree"):
        make_manifest(tmp_path, REPO, "v0.2.0")


def test_local_ubuntu_wheels_are_not_release_artifacts(tmp_path):
    wheel_set(tmp_path)
    path = next(tmp_path.glob("*manylinux*.whl"))
    path.rename(path.with_name(path.name.replace("manylinux_2_27_x86_64.manylinux_2_28_x86_64", "manylinux_2_39_x86_64")))
    with pytest.raises(ValueError, match="manylinux_2_28 or older"):
        make_manifest(tmp_path, REPO, "v0.2.0")


def test_interrupted_upload_is_not_indexed(tmp_path):
    wheel_set(tmp_path)
    github = FakeGitHub()
    path = next(tmp_path.glob("*.whl"))
    github.upload(github.release, path)
    assert not list(release_manifests(github))
    with pytest.raises(ValueError, match="empty deployment"):
        build_index([], REPO, tmp_path / "site")


def test_missing_or_replaced_asset_blocks_index(tmp_path):
    wheel_set(tmp_path)
    github = FakeGitHub()
    publish(github, tmp_path, "v0.2.0")
    name = next(name for name in github.assets if name.endswith(".whl"))
    original = github.assets.pop(name)
    with pytest.raises(ValueError, match="Missing/incomplete"):
        list(release_manifests(github))
    github.assets[name] = b"x" * len(original)
    with pytest.raises(ValueError, match="digest mismatch"):
        list(release_manifests(github))


class Links(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.links = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))


def test_index_preserves_versions_and_channels_with_hashes(tmp_path):
    older = make_manifest(wheel_set(tmp_path / "old"), REPO, "v0.2.0")
    newer = make_manifest(wheel_set(tmp_path / "new", "0.3.0rc1+cu129"), REPO, "v0.3.0rc1")
    other = copy.deepcopy(older)
    other["channel"] = "cu130"
    other["version"] = "0.2.0+cu130"
    for wheel in other["wheels"]:
        wheel["filename"] = wheel["filename"].replace("cu129", "cu130")
    site = tmp_path / "site"
    build_index([newer, other, older], REPO, site)
    links = Links((site / "whl/cu129/c-spikes/index.html").read_text()).links
    assert len(links) == 16
    first = older["wheels"][0]
    assert links[0]["href"] == asset_url(REPO, "v0.2.0", first["filename"]) + "#sha256=" + first["sha256"]
    assert "%2Bcu129" in links[0]["href"]
    assert all(link["data-requires-python"] == ">=3.11" for link in links)
    assert (site / "whl/cu130/c-spikes/index.html").exists()
    assert len(Links((site / "index.html").read_text()).links) == 2


@pytest.mark.parametrize("mutation", ["path", "duplicate", "repository", "hash"])
def test_invalid_manifests_rejected(tmp_path, mutation):
    manifest = make_manifest(wheel_set(tmp_path), REPO, "v0.2.0")
    if mutation == "path":
        manifest["wheels"][0]["filename"] = "../" + manifest["wheels"][0]["filename"]
    elif mutation == "duplicate":
        manifest["wheels"].append(manifest["wheels"][0])
    elif mutation == "repository":
        manifest["repository"] = "unexpected/repository"
    else:
        manifest["wheels"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        validate_manifest(manifest, REPO, "v0.2.0")


def test_pagination_keeps_old_releases(monkeypatch):
    github = GitHub(REPO)
    calls = []
    def request(url):
        calls.append(url)
        return list(range(100)) if url.endswith("page=1") else [100]
    monkeypatch.setattr(github, "request", request)
    assert list(github.pages("releases")) == list(range(101))
    assert len(calls) == 2


def test_index_keeps_old_python_matrices(tmp_path):
    older = make_manifest(wheel_set(tmp_path / "old"), REPO, "v0.2.0")
    older["python_tags"].remove("cp314")
    older["wheels"] = [w for w in older["wheels"] if "-cp314-" not in w["filename"]]
    newer = make_manifest(wheel_set(tmp_path / "new", "0.3.0+cu129"), REPO, "v0.3.0")
    build_index([older, newer], REPO, tmp_path / "site")
    assert len(Links((tmp_path / "site/whl/cu129/c-spikes/index.html").read_text()).links) == 14


@pytest.mark.parametrize("tag", ["main", "v1.0.0+cu129", "v1.0.0.dev1", "v01.0.0"])
def test_invalid_release_tags(tag):
    with pytest.raises(ValueError):
        public_version(tag)


def test_cuda_version_and_baseline(monkeypatch):
    assert wheel_version("0.2.0", cuda=True) == "0.2.0+cu129"
    assert wheel_version("0.2.0") == "0.2.0"
    assert cuda_settings("PORTABLE")["cmake.define.Kokkos_ARCH_TURING75"] == "ON"
    assert cuda_settings("BLACKWELL120")["cmake.define.CMAKE_CUDA_ARCHITECTURES"] == "120"
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_C_SPIKES", "0.2.0")
    assert wheel_version(cuda=True) == "0.2.0+cu129"
