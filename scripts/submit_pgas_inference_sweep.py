#!/usr/bin/env python3
"""Submit one build-once/run-many PGAS inference sweep job."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _script_dir().parents[0]


def _default_matrix_path() -> Path:
    return _script_dir() / "pgas_commit_builds.json"


def _default_sweep_path() -> Path:
    return _script_dir() / "pgas_inference_sweep.json"


def _default_template_path() -> Path:
    return _script_dir() / "pgas_sbatch_template.sbatch"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def _features(item: dict) -> str:
    raw = item.get("features", [])
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(part).strip() for part in raw]
    else:
        values = []
    return ",".join(value for value in values if value)


def _select_build(matrix_path: Path, run_tag: str) -> dict:
    payload = _load_json(matrix_path)
    builds = payload.get("builds")
    if not isinstance(builds, list):
        raise ValueError(f"Matrix has no builds list: {matrix_path}")
    matches = [item for item in builds if isinstance(item, dict) and item.get("run_tag") == run_tag]
    if not matches:
        raise ValueError(f"Build run_tag not found in matrix: {run_tag}")
    return matches[0]


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


def _parse_export_env(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        token = str(value).strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--export-env must be KEY=VALUE, got {token!r}")
        out.append(token)
    return out


def _join_tags(values: Sequence[str] | None) -> str:
    return ":".join(str(value).strip() for value in values or [] if str(value).strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", type=Path, default=_default_sweep_path(), help="Sweep config JSON.")
    parser.add_argument("--matrix", type=Path, default=_default_matrix_path(), help="Commit matrix JSON.")
    parser.add_argument("--template", type=Path, default=_default_template_path(), help="Sbatch template.")
    parser.add_argument("--build-run-tag", help="Build run_tag from matrix. Defaults to sweep JSON build_run_tag.")
    parser.add_argument("--job-run-tag", help="Internal build job run tag. Defaults to sweep_<build-run-tag>.")
    parser.add_argument("--sweep-base-run-tag", help="Override the sweep base_run_tag.")
    parser.add_argument(
        "--include-config-tag",
        action="append",
        default=None,
        help="Run only this config tag from the sweep JSON. Repeatable.",
    )
    parser.add_argument(
        "--exclude-config-tag",
        action="append",
        default=None,
        help="Skip this config tag from the sweep JSON. Repeatable.",
    )
    parser.add_argument("--export-env", action="append", help="Extra environment export KEY=VALUE. Repeatable.")
    parser.add_argument("--sbatch-arg", action="append", default=None, help="Extra sbatch argument. Repeatable.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue sweep after failed config.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore reusable builds and rebuild the selected commit.")
    parser.add_argument("--skip-commit-check", action="store_true", help="Skip git commit existence check.")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch command without submitting.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = _repo_root()
    sweep_path = args.sweep_json.resolve()
    sweep = _load_json(sweep_path)
    build_run_tag = str(args.build_run_tag or sweep.get("build_run_tag", "")).strip()
    if not build_run_tag:
        print("[error] Set --build-run-tag or build_run_tag in sweep JSON.", file=sys.stderr)
        return 2

    try:
        build = _select_build(args.matrix, build_run_tag)
        commit = str(build.get("commit", "")).strip()
        _validate_commit_token(commit)
        if not args.skip_commit_check:
            _validate_commit_exists(repo_root, commit)
        feature_flags = _features(build)
        extra_env = _parse_export_env(args.export_env)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    job_run_tag = str(args.job_run_tag or f"sweep_{build_run_tag}").strip()
    export_parts = [
        "ALL",
        f"REPO_ROOT={repo_root}",
        f"SWEEP_JSON={sweep_path}",
    ]
    if args.sweep_base_run_tag:
        export_parts.append(f"SWEEP_BASE_RUN_TAG={args.sweep_base_run_tag}")
    include_config_tags = _join_tags(args.include_config_tag)
    exclude_config_tags = _join_tags(args.exclude_config_tag)
    if include_config_tags:
        export_parts.append(f"SWEEP_INCLUDE_CONFIG_TAGS={include_config_tags}")
    if exclude_config_tags:
        export_parts.append(f"SWEEP_EXCLUDE_CONFIG_TAGS={exclude_config_tags}")
    if args.continue_on_error:
        export_parts.append("SWEEP_CONTINUE_ON_ERROR=1")
    if args.force_rebuild:
        export_parts.append("FORCE_REBUILD=1")
    export_parts.extend(extra_env)

    cmd = [
        "sbatch",
        *(args.sbatch_arg or []),
        "--export=" + ",".join(export_parts),
        str(args.template),
        job_run_tag,
        commit,
        feature_flags,
    ]
    print(f"[sweep] build={build_run_tag} commit={commit[:8]} features={feature_flags or 'none'}")
    print(f"[sweep] sweep_json={sweep_path}")
    print("  " + shlex.join(cmd))
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        print(f"[error] sbatch failed: {stderr or stdout}", file=sys.stderr)
        return proc.returncode
    print(stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
