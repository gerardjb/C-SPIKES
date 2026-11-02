"""Publish CUDA release assets and build a static, hash-checked Python package index.

Only published releases with a complete c-spikes-cu*.json manifest enter the
index. The manifest is uploaded last, after every wheel has been verified.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import html
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from build_native import CUDA_CHANNEL

PYTHONS = {"cp311", "cp312", "cp313", "cp314"}


def public_version(tag):
    if not tag.startswith("v"):
        raise ValueError("Release tags must start with v, for example v0.2.0")
    version = Version(tag[1:])
    if version.local or version.is_devrelease or str(version) != tag[1:]:
        raise ValueError("Use a canonical public release version without a dev/local suffix")
    return str(version)


def wheel_slot(filename, version):
    if Path(filename).name != filename or not re.fullmatch(r"[A-Za-z0-9_.+\-]+\.whl", filename):
        raise ValueError(f"Invalid wheel filename: {filename}")
    name, parsed, build, tags = parse_wheel_filename(filename)
    if name != "c-spikes" or str(parsed) != version or build:
        raise ValueError(f"Unexpected project/version/build tag: {filename}")
    slots = set()
    for tag in tags:
        if (not re.fullmatch(r"cp3\d{2,}", tag.interpreter)
                or int(tag.interpreter[2:]) < 311 or tag.abi != tag.interpreter):
            raise ValueError(f"Unsupported Python/ABI tag: {filename}")
        if tag.platform == "win_amd64":
            system = "windows"
        elif re.fullmatch(r"manylinux_2_\d+_x86_64", tag.platform):
            if int(tag.platform.split("_")[2]) > 28:
                raise ValueError(f"Release wheels require manylinux_2_28 or older: {filename}")
            system = "linux"
        else:
            raise ValueError(f"Not a supported portable wheel: {filename}")
        slots.add((tag.interpreter, system))
    if len(slots) != 1:
        raise ValueError(f"Ambiguous wheel tags: {filename}")
    return slots.pop()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_manifest(directory, repository, tag):
    version = f"{public_version(tag)}+{CUDA_CHANNEL}"
    wheels = []
    for path in sorted(directory.glob("*.whl")):
        wheel_slot(path.name, version)
        with zipfile.ZipFile(path) as package:
            names = package.namelist()
            metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"Expected one METADATA file in {path.name}")
            info = BytesParser().parsebytes(package.read(metadata[0]))
            if canonicalize_name(info["Name"] or "") != "c-spikes" or info["Version"] != version:
                raise ValueError(f"Wheel filename and metadata disagree: {path.name}")
            if info["Requires-Python"] != ">=3.11":
                raise ValueError(f"Unexpected Python requirement: {path.name}")
            for backend in ("cpu", "gpu"):
                if not any(re.search(rf"/pgas_bound_{backend}\.[^/]+\.(so|pyd)$", name) for name in names):
                    raise ValueError(f"Missing {backend} extension: {path.name}")
            if any(Path(name).name.lower().startswith(("nvcuda", "libcuda.", "libcuda-")) for name in names):
                raise ValueError(f"Bundled NVIDIA driver in {path.name}")
        wheels.append({"filename": path.name, "sha256": sha256(path), "size": path.stat().st_size,
                       "requires_python": ">=3.11"})
    manifest = {"schema": 1, "repository": repository, "tag": tag, "channel": CUDA_CHANNEL,
                "version": version, "python_tags": sorted(PYTHONS), "wheels": wheels}
    validate_manifest(manifest, repository, tag)
    return manifest


def validate_manifest(manifest, repository, tag):
    channel = manifest.get("channel", "")
    if not re.fullmatch(r"cu\d+", channel):
        raise ValueError("Invalid CUDA channel")
    version = f"{public_version(tag)}+{channel}"
    if (manifest.get("schema"), manifest.get("repository"), manifest.get("tag"), manifest.get("version")) != (1, repository, tag, version):
        raise ValueError("Release manifest identity mismatch")
    # Record the matrix per release so adding a Python version later does not
    # make previously complete releases disappear from the index.
    pythons = manifest["python_tags"]
    if (not pythons or len(set(pythons)) != len(pythons)
            or any(not re.fullmatch(r"cp3\d{2,}", python) or int(python[2:]) < 311 for python in pythons)):
        raise ValueError("Invalid release Python matrix")
    expected = {(python, system) for python in pythons for system in ("linux", "windows")}
    slots = set()
    for wheel in manifest["wheels"]:
        slot = wheel_slot(wheel["filename"], version)
        if slot in slots:
            raise ValueError(f"Duplicate wheel for {slot}")
        slots.add(slot)
        if not re.fullmatch(r"[a-f0-9]{64}", wheel["sha256"]) or wheel["size"] <= 0:
            raise ValueError("Invalid wheel digest/size")
        if wheel["requires_python"] != ">=3.11":
            raise ValueError("Unexpected Python requirement")
    if slots != expected:
        raise ValueError(f"Incomplete release wheel matrix; missing {sorted(expected - slots)}, extra {sorted(slots - expected)}")


class GitHub:
    def __init__(self, repository):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected owner/repository")
        self.repository = repository
        self.api = f"https://api.github.com/repos/{repository}"

    def request(self, url, *, data=None, content_type="application/json", raw=False):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "c-spikes-wheel-index"}
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if data is not None:
            headers["Content-Type"] = content_type
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120) as response:
            body = response.read()
        return body if raw else json.loads(body)

    def pages(self, path):
        page = 1
        while True:
            items = self.request(f"{self.api}/{path}?per_page=100&page={page}")
            yield from items
            if len(items) < 100:
                return
            page += 1

    def download(self, tag, filename):
        # Public release downloads need no credentials; do not forward a token
        # to GitHub's redirected asset storage host.
        url = asset_url(self.repository, tag, filename)
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()

    def upload(self, release, path):
        url = release["upload_url"].split("{", 1)[0]
        asset = self.request(f"{url}?{urllib.parse.urlencode({'name': path.name})}",
                             data=path.read_bytes(), content_type="application/octet-stream")
        if asset["state"] != "uploaded" or asset["size"] != path.stat().st_size:
            raise ValueError(f"Incomplete uploaded asset: {path.name}")
        if asset.get("digest") and asset["digest"] != "sha256:" + sha256(path):
            raise ValueError(f"Uploaded asset digest mismatch: {path.name}")


def asset_url(repository, tag, filename):
    return f"https://github.com/{repository}/releases/download/{urllib.parse.quote(tag, safe='')}/{urllib.parse.quote(filename, safe='')}"


def publish(github, directory, tag):
    manifest = make_manifest(directory, github.repository, tag)
    release = github.request(f"{github.api}/releases/tags/{urllib.parse.quote(tag, safe='')}")
    if release["draft"]:
        raise ValueError("Publish the GitHub release before uploading/indexing its wheels")
    existing = {asset["name"]: asset for asset in github.pages(f"releases/{release['id']}/assets")}
    manifest_path = directory / f"c-spikes-{CUDA_CHANNEL}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    paths = [directory / wheel["filename"] for wheel in manifest["wheels"]] + [manifest_path]
    # Check *all* collisions before uploading anything. A retry may reuse exact
    # assets, but it can never silently replace already released wheel bytes.
    for path in paths:
        if asset := existing.get(path.name):
            digest = asset.get("digest")
            if not digest:
                digest = "sha256:" + hashlib.sha256(github.download(tag, path.name)).hexdigest()
            if digest != "sha256:" + sha256(path) or asset["size"] != path.stat().st_size:
                raise ValueError(f"Refusing to overwrite existing release asset: {path.name}")
    for path in paths:
        if path.name not in existing:
            github.upload(release, path)
            print(f"Uploaded {path.name}")
        else:
            print(f"Verified existing {path.name}")


def release_manifests(github):
    for release in github.pages("releases"):
        if release["draft"]:
            continue
        assets = {asset["name"]: asset for asset in github.pages(f"releases/{release['id']}/assets")}
        for name in sorted(assets):
            if not re.fullmatch(r"c-spikes-cu\d+\.json", name):
                continue
            manifest = json.loads(github.download(release["tag_name"], name))
            validate_manifest(manifest, github.repository, release["tag_name"])
            if name != f"c-spikes-{manifest['channel']}.json":
                raise ValueError("Manifest filename does not match its channel")
            for wheel in manifest["wheels"]:
                asset = assets.get(wheel["filename"])
                if not asset or asset["state"] != "uploaded" or asset["size"] != wheel["size"]:
                    raise ValueError(f"Missing/incomplete release asset: {wheel['filename']}")
                if asset.get("digest") and asset["digest"] != "sha256:" + wheel["sha256"]:
                    raise ValueError(f"Release asset digest mismatch: {wheel['filename']}")
            yield manifest


def write_page(path, title, links):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                    f"<title>{html.escape(title)}</title></head><body>\n"
                    f"<h1>{html.escape(title)}</h1>\n" + "\n".join(links) + "\n</body></html>\n", encoding="utf-8")


def build_index(manifests, repository, output):
    channels = {}
    seen = {}
    for manifest in sorted(manifests, key=lambda m: Version(m["version"])):
        validate_manifest(manifest, repository, manifest["tag"])
        for wheel in manifest["wheels"]:
            name = wheel["filename"]
            if name in seen:
                raise ValueError(f"Duplicate wheel across releases: {name}")
            seen[name] = wheel["sha256"]
            url = asset_url(repository, manifest["tag"], name) + "#sha256=" + wheel["sha256"]
            link = f'<a href="{html.escape(url, quote=True)}" data-requires-python="{html.escape(wheel["requires_python"], quote=True)}">{html.escape(name)}</a><br>'
            channels.setdefault(manifest["channel"], []).append(link)
    if not channels:
        raise ValueError("No complete published CUDA releases found; refusing an empty deployment")
    for channel, links in sorted(channels.items()):
        write_page(output / "whl" / channel / "c-spikes/index.html", f"c-spikes ({channel})", links)
        write_page(output / "whl" / channel / "index.html", f"CUDA wheels ({channel})",
                   ['<a href="c-spikes/">c-spikes</a>'])
    write_page(output / "index.html", "C-SPIKES CUDA wheels",
               [f'<a href="whl/{channel}/">{channel}</a><br>' for channel in sorted(channels)])
    (output / ".nojekyll").touch()
    print(f"Indexed {len(seen)} wheels across {len(channels)} CUDA channels in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version")
    version.add_argument("--tag", required=True)
    for name in ("manifest", "publish"):
        command = commands.add_parser(name)
        command.add_argument("--repository", required=True)
        command.add_argument("--tag", required=True)
        command.add_argument("--wheel-dir", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--repository", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifests", type=Path, help="Offline preview from local manifest JSON files")
    args = parser.parse_args()
    if args.command == "version":
        print(public_version(args.tag))
    elif args.command == "manifest":
        print(json.dumps(make_manifest(args.wheel_dir, args.repository, args.tag), indent=2))
    elif args.command == "publish":
        publish(GitHub(args.repository), args.wheel_dir, args.tag)
    else:
        manifests = ([json.loads(path.read_text()) for path in sorted(args.manifests.glob("c-spikes-cu*.json"))]
                     if args.manifests else release_manifests(GitHub(args.repository)))
        build_index(manifests, args.repository, args.output)


if __name__ == "__main__":
    main()
