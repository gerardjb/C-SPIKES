#!/usr/bin/env python3
"""
Submit PGAS commit-comparison sbatch jobs from a JSON build matrix.

Example:
  python scripts/submit_pgas_commit_builds.py --dry-run
  python scripts/submit_pgas_commit_builds.py --run-tag-prefix cmp_
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence


def _default_matrix_path() -> Path:
    return Path(__file__).resolve().parent / "pgas_commit_builds.json"


def _default_template_path() -> Path:
    return Path(__file__).resolve().parent / "pgas_sbatch_template.sbatch"


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
        help="Prefix added to each run_tag before submission.",
    )
    parser.add_argument(
        "--run-tag-suffix",
        type=str,
        default="",
        help="Suffix added to each run_tag before submission.",
    )
    parser.add_argument(
        "--include-run-tag",
        action="append",
        default=None,
        help="Submit only these matrix run_tag values (repeatable).",
    )
    parser.add_argument(
        "--exclude-run-tag",
        action="append",
        default=None,
        help="Exclude these matrix run_tag values (repeatable).",
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
        help="Extra argument forwarded to sbatch before the template path (repeatable).",
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
        "--allow-unresolved-template",
        action="store_true",
        help=(
            "Allow template files that still contain placeholder tokens "
            "like <your_path_to_...>."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    return parser.parse_args(argv)


def _load_matrix(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Matrix JSON must be an object.")
    builds = payload.get("builds")
    if not isinstance(builds, list):
        raise ValueError("Matrix JSON must contain a 'builds' list.")
    out: List[dict] = []
    for item in builds:
        if not isinstance(item, dict):
            continue
        run_tag = str(item.get("run_tag", "")).strip()
        commit = str(item.get("commit", "")).strip()
        if not run_tag or not commit:
            continue
        out.append(item)
    if not out:
        raise ValueError("No valid build entries found in matrix.")
    return out


def _filter_builds(
    builds: Iterable[dict],
    include_tags: set[str],
    exclude_tags: set[str],
) -> List[dict]:
    selected: List[dict] = []
    for item in builds:
        run_tag = str(item.get("run_tag", "")).strip()
        if include_tags and run_tag not in include_tags:
            continue
        if run_tag in exclude_tags:
            continue
        selected.append(item)
    return selected


def _validate_template(path: Path, *, allow_unresolved: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    content = path.read_text(encoding="utf-8")
    if allow_unresolved:
        return
    if "<your_path_to_" in content:
        raise ValueError(
            "Template contains unresolved path placeholders. "
            "Edit template paths first or pass --allow-unresolved-template."
        )


def _validate_commit_token(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise ValueError(f"Invalid commit hash token: {commit}")


def _extract_job_id(text: str) -> str | None:
    # Typical sbatch output: "Submitted batch job 12345678"
    match = re.search(r"\bSubmitted batch job\s+(\d+)\b", text)
    if match:
        return match.group(1)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_template(args.template, allow_unresolved=bool(args.allow_unresolved_template))
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
        run_tag = f"{args.run_tag_prefix}{base_run_tag}{args.run_tag_suffix}"
        try:
            _validate_commit_token(commit)
        except ValueError as exc:
            print(f"[error] {base_run_tag}: {exc}", file=sys.stderr)
            failed += 1
            if not args.continue_on_error:
                return 1
            continue

        cmd = ["sbatch", *sbatch_extra, str(args.template), run_tag, commit]
        prefix = f"[{idx}/{len(selected)}] {base_run_tag} -> {run_tag} ({commit[:8]})"
        if label:
            prefix = f"{prefix} {label}"
        print(prefix)
        print("  " + " ".join(cmd))

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

