"""Validate a tagged PyPI bundle; never build, execute package code, or publish."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import zipfile

PYTHONS = {"cp311", "cp312", "cp313", "cp314"}
NUMBER = r"(?:0|[1-9]\d*)"
TAG = re.compile(rf"v({NUMBER}\.{NUMBER}\.{NUMBER}(?:(?:a|b|rc){NUMBER})?(?:\.post{NUMBER})?)\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def version_for_tag(tag):
    match = TAG.fullmatch(tag)
    require(match is not None, "Use a canonical release tag such as v0.2.0 or v0.2.0rc1 (no dev/local suffix)")
    return match[1]


def check_metadata(data, version):
    info = BytesParser().parsebytes(data)
    require(re.sub(r"[-_.]+", "-", info.get("Name", "").lower()) == "c-spikes", "Wrong distribution name")
    require(info.get("Version") == version, "Metadata version differs from release tag")
    require(info.get("Requires-Python") == ">=3.11", "Unexpected Python requirement")


def validate_bundle(directory, version):
    files = sorted(Path(directory).iterdir())
    require(len(files) == 9 and all(p.is_file() and not p.is_symlink() for p in files),
            "Require exactly eight wheels and one sdist")
    seen, sdists, hashes = set(), 0, {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.suffix == ".whl":
            parts = path.name[:-4].split("-")
            require(len(parts) == 5 and parts[0] == "c_spikes" and parts[1] == version,
                    "Unexpected wheel project/version/build tag")
            _, _, python, abi, platform = parts
            require(python in PYTHONS and abi == python, "Unexpected Python/ABI tag")
            if platform == "win_amd64":
                system, suffix = "Windows", "pyd"
            else:
                for tag in platform.split("."):
                    match = re.fullmatch(r"manylinux_2_(\d+)_x86_64", tag)
                    require(match is not None and int(match[1]) <= 28,
                            "Require x86-64 manylinux_2_28 or older wheels")
                system, suffix = "Linux", "so"
            require((system, python) not in seen, "Duplicate platform/Python wheel")
            seen.add((system, python))
            with zipfile.ZipFile(path) as wheel:
                names = wheel.namelist()
                require(len(names) == len(set(names)), "Duplicate wheel member")
                metas = [n for n in names if n.endswith(".dist-info/METADATA")]
                require(metas == [f"c_spikes-{version}.dist-info/METADATA"], "Unexpected wheel metadata path")
                check_metadata(wheel.read(metas[0]), version)
                for backend in ("cpu", "gpu"):
                    native = [n for n in names if re.fullmatch(
                        rf"c_spikes/pgas/pgas_bound_{backend}\.[^/]+\.{suffix}", n)]
                    require(len(native) == 1, f"Missing/duplicate {backend} extension")
                require("c_spikes/pgas/backends.py" in names, "Missing lazy backend loader")
                require(not any(Path(n).name.lower().startswith(("nvcuda", "libcuda.", "libcuda-")) for n in names),
                        "Bundled NVIDIA driver")
        elif path.name == f"c_spikes-{version}.tar.gz":
            sdists += 1
            with tarfile.open(path, "r:gz") as archive:
                matches = [m for m in archive.getmembers()
                           if m.name == f"c_spikes-{version}/PKG-INFO" and m.isfile()]
                require(len(matches) == 1, "Missing/duplicate sdist metadata")
                check_metadata(archive.extractfile(matches[0]).read(), version)
        else:
            raise ValueError("Unexpected distribution: " + path.name)
    require(seen == {(s, p) for s in ("Linux", "Windows") for p in PYTHONS} and sdists == 1,
            "Incomplete release matrix")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("version", "bundle"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--directory", type=Path, default=Path("dist/release"))
    parser.add_argument("--manifest", type=Path, default=Path("release-manifest.json"))
    args = parser.parse_args()
    version = version_for_tag(args.tag)
    if args.command == "version":
        print(version)
        return
    receipt = {"tag": args.tag, "version": version, "backend": "cpu+cuda",
               "source_sha": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
               "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
               "files": validate_bundle(args.directory, version)}
    args.manifest.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
