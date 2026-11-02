"""Fail-closed admission for existing CI artifacts; never builds or publishes."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from urllib.request import Request, urlopen
import zipfile

REPOSITORY = "gerardjb/C-SPIKES"
TAG = re.compile(r"v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)\Z")
PYTHONS = {"cp311", "cp312", "cp313", "cp314"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def version_for_tag(tag):
    match = TAG.fullmatch(tag)
    require(match is not None, "Use a release tag such as v0.1.2 or v0.1.2rc1")
    version = match[1]
    require(all(str(int(n)) == n for n in version.split("rc")[0].split("a")[0].split("b")[0].split(".")),
            "Release version must not contain leading zeroes")
    return version


def api(path):
    request = Request("https://api.github.com/repos/" + REPOSITORY + "/" + path,
                      headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"],
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def admit_run(run, tag, sha):
    require(run["repository"]["full_name"] == REPOSITORY and
            run["head_repository"]["full_name"] == REPOSITORY, "Wrong source repository")
    require(run["path"] == ".github/workflows/build.yml", "Wrong build workflow")
    require(run["event"] == "push" and run["head_branch"] == tag, "Require a tag-push build, not PR/main artifacts")
    require(run["head_sha"] == sha, "Build SHA does not match the current tag")
    require(run["status"] == "completed" and run["conclusion"] == "success", "Build must be fully successful")


def select_artifacts(artifacts, backend):
    require(backend in {"cpu", "cuda"}, "Select exactly one backend")
    names = [f"wheels-Linux-{backend}", f"wheels-Windows-{backend}", "sdist"]
    selected = []
    for name in names:
        matches = [a for a in artifacts if a["name"] == name]
        require(len(matches) == 1, "Missing or duplicate artifact: " + name)
        require(not matches[0]["expired"], "Expired artifact: " + name)
        selected.append(matches[0])
    return selected


def source(args):
    version = version_for_tag(args.tag)
    require(re.fullmatch(r"[1-9]\d*", args.run_id) is not None, "Invalid build run ID")
    ref = api("git/ref/tags/" + args.tag)["object"]
    for _ in range(5):
        if ref["type"] != "tag":
            break
        ref = api("git/tags/" + ref["sha"])["object"]
    require(ref["type"] == "commit", "Tag must resolve to a commit")
    sha = ref["sha"]
    comparison = api("compare/main..." + sha)
    require(comparison["status"] in {"behind", "identical"}, "Tagged source must belong to main history")
    run = api("actions/runs/" + args.run_id)
    admit_run(run, args.tag, sha)
    artifacts = []
    for page in range(1, 101):
        batch = api(f"actions/runs/{args.run_id}/artifacts?per_page=100&page={page}")["artifacts"]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ValueError("Unexpected artifact pagination")
    selected = select_artifacts(artifacts, args.backend)
    receipt = {"repository": REPOSITORY, "tag": args.tag, "version": version,
               "source_sha": sha, "run_id": args.run_id, "run_attempt": run["run_attempt"],
               "backend": args.backend,
               "artifacts": [{k: a.get(k) for k in ("id", "name", "digest", "size_in_bytes", "created_at")} for a in selected]}
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("artifact_ids=" + ",".join(str(a["id"]) for a in selected) + "\n")
    print(json.dumps(receipt, indent=2))


def metadata(data, version):
    message = BytesParser().parsebytes(data)
    require(re.sub(r"[-_.]+", "-", message.get("Name", "").lower()) == "c-spikes", "Wrong distribution name")
    require(message.get("Version") == version, "Distribution version differs from release tag")


def validate_bundle(directory, version, backend):
    require(backend in {"cpu", "cuda"}, "Select exactly one backend")
    files = sorted(Path(directory).iterdir())
    require(len(files) == 9 and all(p.is_file() for p in files), "Require exactly eight wheels and one sdist")
    seen, sdists, hashes = set(), 0, {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.name.endswith(".whl"):
            parts = path.name[:-4].split("-")
            require(len(parts) == 5 and parts[0] == "c_spikes" and parts[1] == version, "Unexpected wheel filename")
            _, _, python, abi, platform = parts
            require(python in PYTHONS and abi == python, "Unexpected Python/ABI tag")
            if platform == "win_amd64":
                system = "Windows"
            else:
                require(all(re.fullmatch(r"manylinux_\d+_\d+_x86_64", t) for t in platform.split(".")), "Require repaired x86-64 manylinux wheels")
                system = "Linux"
            require((system, python) not in seen, "Duplicate platform/Python wheel")
            seen.add((system, python))
            with zipfile.ZipFile(path) as wheel:
                names = wheel.namelist()
                require(len(names) == len(set(names)), "Duplicate wheel member")
                metas = [n for n in names if n.endswith(".dist-info/METADATA")]
                require(len(metas) == 1, "Missing/duplicate wheel metadata")
                metadata(wheel.read(metas[0]), version)
                native = [Path(n).name for n in names if n.startswith("c_spikes/pgas/") and n.endswith((".so", ".pyd"))]
                require(any(n.startswith("pgas_bound_cpu.") for n in native), "Missing CPU extension")
                require(any(n.startswith("pgas_bound_gpu.") for n in native) == (backend == "cuda"), "Wheel backend differs from selection")
        elif path.name == f"c_spikes-{version}.tar.gz":
            sdists += 1
            with tarfile.open(path, "r:gz") as archive:
                matches = [m for m in archive.getmembers() if m.name == f"c_spikes-{version}/PKG-INFO" and m.isfile()]
                require(len(matches) == 1, "Missing/duplicate sdist metadata")
                metadata(archive.extractfile(matches[0]).read(), version)
        else:
            raise ValueError("Unexpected distribution: " + path.name)
    require(seen == {(s, p) for s in ("Linux", "Windows") for p in PYTHONS} and sdists == 1, "Incomplete release matrix")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    src = sub.add_parser("source")
    src.add_argument("--run-id", required=True)
    src.add_argument("--tag", required=True)
    src.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    src.add_argument("--receipt", default="source.json")
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--directory", default="dist")
    bundle.add_argument("--receipt", default="source.json")
    args = parser.parse_args()
    if args.command == "source":
        source(args)
    else:
        receipt = json.loads(Path(args.receipt).read_text())
        receipt["files"] = validate_bundle(args.directory, receipt["version"], receipt["backend"])
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
