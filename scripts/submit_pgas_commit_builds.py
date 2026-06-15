#!/usr/bin/env python3
"""Submit PGAS Allan-ladder commit-comparison sbatch jobs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


def _default_matrix_path() -> Path:
    return Path(__file__).resolve().parent / "pgas_commit_builds.json"


def _default_template_path() -> Path:
    return Path(__file__).resolve().parent / "pgas_sbatch_template.sbatch"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=_default_matrix_path(),
        help="Path to commit/run-tag matrix JSON.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=_default_template_path(),
        help="Path to sbatch template script.",
    )
    parser.add_argument(
        "--run-tag-prefix",
        type=str,
        default="",
        help="Prefix added to each matrix run_tag before submission.",
    )
    parser.add_argument(
        "--run-tag-suffix",
        type=str,
        default="",
        help="Suffix added to each matrix run_tag before submission.",
    )
    parser.add_argument(
        "--include-run-tag",
        action="append",
        default=None,
        help="Submit only these matrix run_tag values. Repeatable.",
    )
    parser.add_argument(
        "--exclude-run-tag",
        action="append",
        default=None,
        help="Exclude these matrix run_tag values. Repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of jobs to submit after filtering.",
    )
    parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=None,
        help="Extra argument forwarded to sbatch before the template path. Repeatable.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between submissions.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue submitting if one job fails.",
    )
    parser.add_argument(
        "--skip-commit-check",
        action="store_true",
        help="Do not verify that matrix commits exist in the current repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    return parser.parse_args(argv)


def _load_matrix(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Matrix JSON must be an object.")
    builds = payload.get("builds")
    if not isinstance(builds, list):
        raise ValueError("Matrix JSON must contain a 'builds' list.")

    valid_builds: list[dict] = []
    seen_tags: set[str] = set()
    for item in builds:
        if not isinstance(item, dict):
            continue
        run_tag = str(item.get("run_tag", "")).strip()
        commit = str(item.get("commit", "")).strip()
        if not run_tag or not commit:
            continue
        if run_tag in seen_tags:
            raise ValueError(f"Duplicate run_tag in matrix: {run_tag}")
        seen_tags.add(run_tag)
        valid_builds.append(item)

    if not valid_builds:
        raise ValueError("No valid build entries found in matrix.")
    return valid_builds


def _filter_builds(
    builds: Iterable[dict],
    include_tags: set[str],
    exclude_tags: set[str],
) -> list[dict]:
    selected: list[dict] = []
    for item in builds:
        run_tag = str(item.get("run_tag", "")).strip()
        if include_tags and run_tag not in include_tags:
            continue
        if run_tag in exclude_tags:
            continue
        selected.append(item)
    return selected


def _validate_template(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _validate_commit_token(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise ValueError(f"Invalid commit hash token: {commit}")


def _validate_commit_exists(repo_root: Path, commit: str) -> None:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"Commit is not available in this repository: {commit}")


def _normalize_features(item: dict) -> str:
    raw = item.get("features", [])
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(part).strip() for part in raw]
    else:
        values = []
    values = [value for value in values if value]
    return ",".join(values)


def _extract_job_id(text: str) -> str | None:
    match = re.search(r"\bSubmitted batch job\s+(\d+)\b", text)
    if match:
        return match.group(1)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = _repo_root()

    try:
        _validate_template(args.template)
        builds = _load_matrix(args.matrix)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    include_tags = {str(x).strip() for x in (args.include_run_tag or []) if str(x).strip()}
    exclude_tags = {str(x).strip() for x in (args.exclude_run_tag or []) if str(x).strip()}
    selected = _filter_builds(builds, include_tags, exclude_tags)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    if not selected:
        print("[info] No builds selected after filtering.")
        return 0

    sbatch_extra = list(args.sbatch_arg or [])
    submitted = 0
    failed = 0

    for idx, item in enumerate(selected, start=1):
        base_run_tag = str(item["run_tag"]).strip()
        commit = str(item["commit"]).strip()
        label = str(item.get("label", "")).strip()
        covers = item.get("covers", [])
        features = _normalize_features(item)
        run_tag = f"{args.run_tag_prefix}{base_run_tag}{args.run_tag_suffix}"

        try:
            _validate_commit_token(commit)
            if not args.skip_commit_check:
                _validate_commit_exists(repo_root, commit)
        except ValueError as exc:
            print(f"[error] {base_run_tag}: {exc}", file=sys.stderr)
            failed += 1
            if not args.continue_on_error:
                return 1
            continue

        cmd = ["sbatch", *sbatch_extra, str(args.template), run_tag, commit, features]
        prefix = f"[{idx}/{len(selected)}] {base_run_tag} -> {run_tag} ({commit[:8]})"
        if label:
            prefix = f"{prefix} {label}"
        print(prefix)
        if covers:
            print(f"  covers: {', '.join(str(item) for item in covers)}")
        if features:
            print(f"  features: {features}")
        print("  " + shlex.join(cmd))

        if args.dry_run:
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            failed += 1
            print(f"[error] sbatch failed for {run_tag}: {stderr or stdout}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
        else:
            submitted += 1
            job_id = _extract_job_id(stdout)
            if job_id is not None:
                print(f"  submitted job_id={job_id}")
            else:
                print(f"  submitted: {stdout}")

        if args.sleep_seconds > 0 and idx < len(selected):
            time.sleep(float(args.sleep_seconds))

    if args.dry_run:
        print(f"[done] dry-run commands: {len(selected)}")
        return 0

    print(f"[done] submitted={submitted} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
