#!/usr/bin/env python3
"""Run multiple PGAS inference configurations from one built commit environment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def _optional(value: Any) -> Any:
    if value is None:
        return None
    token = str(value).strip()
    if token.lower() in {"", "none", "null"}:
        return None
    return value


def _resolve_path(value: str | Path | None, *, default: Path, repo_root: Path) -> Path:
    raw = default if value is None else Path(value)
    raw = raw.expanduser()
    if raw.is_absolute():
        return raw
    return repo_root / raw


def _parse_dataset_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part for part in raw.split() if part]
    if isinstance(raw, list):
        return [str(part).strip() for part in raw if str(part).strip()]
    raise ValueError("dataset_tags must be a list or whitespace-separated string.")


def _read_trial_selection_keys(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Trial selection JSON must be an object: {path}")
    return sorted(str(key).strip() for key in payload.keys() if str(key).strip())


def _write_first_trial_selection(path: Path, dataset_tags: Iterable[str]) -> None:
    payload = {str(tag): [0] for tag in dataset_tags if str(tag).strip()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _feature_set(raw: str | Sequence[str] | None) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {part.strip() for part in raw.split(",") if part.strip()}
    return {str(part).strip() for part in raw if str(part).strip()}


def _tag_set(raw: Sequence[str] | None) -> set[str]:
    return {str(part).strip() for part in raw or [] if str(part).strip()}


def _config_tag(item: dict[str, Any], idx: int) -> str:
    return str(item.get("tag", f"cfg{idx:03d}")).strip()


def _filter_configs(
    configs: Sequence[Any],
    *,
    include_tags: set[str],
    exclude_tags: set[str],
) -> list[tuple[int, dict[str, Any], str]]:
    selected: list[tuple[int, dict[str, Any], str]] = []
    seen_tags: set[str] = set()
    for idx, item in enumerate(configs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Config {idx} must be an object.")
        tag = _config_tag(item, idx)
        if not tag:
            raise ValueError(f"Config {idx} has an empty tag.")
        if tag in seen_tags:
            raise ValueError(f"Duplicate config tag: {tag}")
        seen_tags.add(tag)
        if include_tags and tag not in include_tags:
            continue
        if tag in exclude_tags:
            continue
        selected.append((idx, item, tag))

    if include_tags:
        missing = sorted(include_tags - seen_tags)
        if missing:
            raise ValueError(f"Included config tag(s) not found: {', '.join(missing)}")
    return selected


def _merged_config(sweep: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    merged = {
        key: value
        for key, value in sweep.items()
        if key not in {"configs", "schema_version", "description"}
    }
    merged.update(item)
    return merged


def _add_if_present(cmd: list[str], flag: str, value: Any) -> None:
    value = _optional(value)
    if value is not None:
        cmd.extend([flag, str(value)])


def _build_cli_command(
    *,
    python_executable: str,
    config: dict[str, Any],
    run_tag: str,
    run_root: Path,
    repo_root: Path,
    data_root: Path,
    pgas_constants: Path,
    pgas_gparam: Path,
    features: set[str],
    dry_run: bool,
) -> tuple[list[str], Path | None]:
    cache_root = run_root / "inference_cache"
    pgas_output_root = run_root / "pgas_output"
    eval_root = run_root / "cli_evaluation"
    slurm_root = run_root / "slurm"

    dataset_tags = _parse_dataset_tags(config.get("dataset_tags"))
    trial_selection_path_raw = _optional(config.get("trial_selection_path"))
    trial_selection_path = (
        _resolve_path(trial_selection_path_raw, default=Path(), repo_root=repo_root)
        if trial_selection_path_raw is not None
        else None
    )
    if not dataset_tags and trial_selection_path is not None:
        dataset_tags = _read_trial_selection_keys(trial_selection_path)
    if not dataset_tags:
        raise ValueError(f"{run_tag}: set dataset_tags or trial_selection_path.")

    first_trial_only = _as_bool(config.get("first_trial_only"), default=False)
    effective_trial_selection_path = trial_selection_path
    generated_selection_path: Path | None = None
    if first_trial_only and effective_trial_selection_path is None:
        generated_selection_path = slurm_root / "first_trial_selection.json"
        effective_trial_selection_path = generated_selection_path
        if not dry_run:
            _write_first_trial_selection(generated_selection_path, dataset_tags)

    edges_path_raw = _optional(config.get("edges_path"))
    if edges_path_raw is None:
        # Always pass a run-local nonexistent path so c_spikes.cli.run does not fall back to
        # its default results/excitatory_time_stamp_edges.npy if that file exists.
        edges_path = slurm_root / "edges_disabled.npy"
    else:
        edges_path = _resolve_path(edges_path_raw, default=Path(), repo_root=repo_root)

    cmd = [
        python_executable,
        "-m",
        "c_spikes.cli.run",
        "--data-root",
        str(data_root),
        "--method",
        "pgas",
        "--smoothing-level",
        str(config.get("smoothing_level", "raw")),
        "--pgas-constants",
        str(pgas_constants),
        "--pgas-gparam",
        str(pgas_gparam),
        "--pgas-output-root",
        str(pgas_output_root),
        "--output-root",
        str(eval_root),
        "--cache-root",
        str(cache_root),
        "--run-tag",
        run_tag,
        "--pgas-bm-sigma",
        str(config.get("bm_sigma", "auto")),
        "--bm-sigma-spike-gap",
        str(config.get("bm_sigma_spike_gap", 0.15)),
        "--corr-sigma-ms",
        str(config.get("corr_sigma_ms", 50.0)),
        "--edges-path",
        str(edges_path),
    ]

    if "bm_bounds" in features:
        _add_if_present(cmd, "--pgas-bm-sigma-min", config.get("bm_sigma_min", 0.0005))
        _add_if_present(cmd, "--pgas-bm-sigma-max", config.get("bm_sigma_max"))
    elif config.get("bm_sigma_max") is not None:
        print(f"[warn] {run_tag}: commit does not expose bm bounds; ignoring bm_sigma_max.")

    if "low_activity_mask" in features:
        if _as_bool(config.get("bm_sigma_use_low_activity_mask"), default=False):
            cmd.append("--pgas-bm-sigma-use-low-activity-mask")
    elif _as_bool(config.get("bm_sigma_use_low_activity_mask"), default=False):
        print(f"[warn] {run_tag}: commit does not expose low-activity mask; ignoring.")

    if "sigma2_prior" in features:
        _add_if_present(cmd, "--pgas-sigma2-target", config.get("sigma2_target"))
        _add_if_present(cmd, "--pgas-sigma2-alpha", config.get("sigma2_alpha"))
        if config.get("sigma2_target") is not None:
            _add_if_present(
                cmd,
                "--pgas-sigma2-prior-strength",
                config.get("sigma2_prior_strength", 4),
            )
    elif config.get("sigma2_target") is not None or config.get("sigma2_alpha") is not None:
        print(f"[warn] {run_tag}: commit does not expose sigma2 prior knobs; ignoring.")

    _add_if_present(cmd, "--pgas-resample-fs", config.get("pgas_resample_fs"))
    _add_if_present(cmd, "--pgas-maxspikes", config.get("pgas_maxspikes"))
    if _as_bool(config.get("pgas_c0_first_y"), default=False):
        cmd.append("--pgas-c0-first-y")
    if first_trial_only:
        cmd.append("--first-trial-only")
    if _as_bool(config.get("trialwise_correlations"), default=True):
        cmd.append("--trialwise-correlations")
    if _as_bool(config.get("use_cache"), default=False):
        cmd.append("--use-cache")
    if effective_trial_selection_path is not None:
        cmd.extend(["--trial-selection-path", str(effective_trial_selection_path)])
    for dataset_tag in dataset_tags:
        cmd.extend(["--dataset", dataset_tag])
    return cmd, generated_selection_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", type=Path, required=True, help="Sweep config JSON.")
    parser.add_argument("--repo-root", type=Path, default=_repo_root(), help="Repository root.")
    parser.add_argument("--data-root", type=Path, help="Data root. Defaults to repo sample data.")
    parser.add_argument("--results-parent", type=Path, help="Run output parent directory.")
    parser.add_argument("--pgas-constants", type=Path, help="Base PGAS constants JSON.")
    parser.add_argument("--pgas-gparam", type=Path, help="PGAS gparam file.")
    parser.add_argument("--features", type=str, default="", help="Comma-separated commit feature flags.")
    parser.add_argument("--base-run-tag", type=str, help="Override sweep base_run_tag.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for c_spikes CLI.")
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
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running inference.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    sweep_path = args.sweep_json.resolve()
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(sweep, dict):
        raise ValueError("Sweep JSON must be an object.")
    configs = sweep.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("Sweep JSON must contain a non-empty configs list.")

    data_root = _resolve_path(
        args.data_root,
        default=Path(sweep.get("data_root", "sample_data/janelia_8f/excitatory")),
        repo_root=repo_root,
    )
    results_parent = _resolve_path(
        args.results_parent,
        default=Path(sweep.get("results_parent", data_root / "spike_inference")),
        repo_root=repo_root,
    )
    pgas_constants = _resolve_path(
        args.pgas_constants,
        default=Path(sweep.get("pgas_constants", "parameter_files/constants_GCaMP8_soma.json")),
        repo_root=repo_root,
    )
    pgas_gparam = _resolve_path(
        args.pgas_gparam,
        default=Path(sweep.get("pgas_gparam", "src/c_spikes/pgas/20230525_gold.dat")),
        repo_root=repo_root,
    )
    base_run_tag = str(args.base_run_tag or sweep.get("base_run_tag") or sweep_path.stem).strip()
    if not base_run_tag:
        raise ValueError("Set base_run_tag in JSON or pass --base-run-tag.")
    features = _feature_set(args.features or sweep.get("features"))
    include_tags = _tag_set(args.include_config_tag)
    exclude_tags = _tag_set(args.exclude_config_tag)
    selected_configs = _filter_configs(
        configs,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )
    if not selected_configs:
        raise ValueError("No configs selected after include/exclude filtering.")

    print(f"[sweep] sweep_json={sweep_path}")
    print(f"[sweep] base_run_tag={base_run_tag}")
    print(f"[sweep] features={','.join(sorted(features)) or 'none'}")
    if include_tags:
        print(f"[sweep] include_config_tags={','.join(sorted(include_tags))}")
    if exclude_tags:
        print(f"[sweep] exclude_config_tags={','.join(sorted(exclude_tags))}")
    print(f"[sweep] configs={len(selected_configs)} of {len(configs)}")

    failed = 0
    for out_idx, (idx, item, tag) in enumerate(selected_configs, start=1):
        run_tag = f"{base_run_tag}_{tag}"
        run_root = results_parent / run_tag
        config = _merged_config(sweep, item)
        cmd, generated_selection_path = _build_cli_command(
            python_executable=args.python_executable,
            config=config,
            run_tag=run_tag,
            run_root=run_root,
            repo_root=repo_root,
            data_root=data_root,
            pgas_constants=pgas_constants,
            pgas_gparam=pgas_gparam,
            features=features,
            dry_run=bool(args.dry_run),
        )
        print(f"[{out_idx}/{len(selected_configs)}] {run_tag}")
        print("  " + shlex.join(cmd))
        if args.dry_run:
            continue

        cache_constants = run_root / "inference_cache" / "pgas_constants"
        if cache_constants.exists():
            import shutil

            shutil.rmtree(cache_constants)
        (run_root / "slurm").mkdir(parents=True, exist_ok=True)
        if generated_selection_path is not None and not generated_selection_path.exists():
            dataset_tags = _parse_dataset_tags(config.get("dataset_tags"))
            _write_first_trial_selection(generated_selection_path, dataset_tags)

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sweep_json": str(sweep_path),
            "base_run_tag": base_run_tag,
            "run_tag": run_tag,
            "features": sorted(features),
            "config": config,
            "command": cmd,
        }
        (run_root / "slurm" / "sweep_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failed += 1
            print(f"[error] {run_tag} failed with exit code {proc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return proc.returncode

    if args.dry_run:
        print(f"[done] dry-run configs={len(selected_configs)}")
        return 0
    print(f"[done] configs={len(selected_configs)} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
