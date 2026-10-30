"""Unpack hash-pinned NVIDIA compiler/runtime components without installing a driver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "build/tools/cuda")
    parser.add_argument("--platform", choices=("linux-x86_64", "windows-x86_64"),
                        default=platform.system().lower() + "-x86_64")
    args = parser.parse_args()
    manifest_path = ROOT / "cmake/cuda-12.6.3.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    destination = args.destination.resolve()
    receipt = destination / "c-spikes-cuda.json"
    identity = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "platform": args.platform, "layout_version": 2}
    if receipt.exists() and json.loads(receipt.read_text()) == identity:
        print(destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination.parent / "downloads"
    cache.mkdir(exist_ok=True)
    for name, platforms in manifest["components"].items():
        component = platforms[args.platform]
        archive = cache / Path(component["relative_path"]).name
        if not archive.exists():
            url = "https://developer.download.nvidia.com/compute/cuda/redist/" + component["relative_path"]
            print(f"Downloading {name} from {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with archive.open("rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != component["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {archive}; remove this download and retry")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            extracted = Path(temporary)
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as package:
                    package.extractall(extracted)
            else:
                with tarfile.open(archive) as package:
                    package.extractall(extracted, filter="data")
            entries = list(extracted.iterdir())
            if len(entries) != 1 or not entries[0].is_dir():
                raise RuntimeError(f"Unexpected NVIDIA archive layout: {archive}")
            shutil.copytree(entries[0], destination, dirs_exist_ok=True)
            notices = destination / "licenses" / name
            notices.mkdir(parents=True, exist_ok=True)
            for license_file in entries[0].glob("LICENSE*"):
                shutil.copy2(license_file, notices / license_file.name)
    # Linux component archives use lib/, while nvcc's toolkit layout expects
    # lib64/. Keep both CMake discovery and nvcc's own linker search working.
    if args.platform == "linux-x86_64" and not (destination / "lib64").exists():
        (destination / "lib64").symlink_to("lib", target_is_directory=True)
    receipt.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
