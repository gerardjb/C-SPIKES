from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from c_spikes.gui.smc_viz import cache_tag_to_epoch_id


def _looks_like_pgas_cache_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_dir():
            continue
        has_json = any(child.glob("*.json"))
        has_mat = any(child.glob("*.mat"))
        if has_json and has_mat:
            return True
    return False


def _resolve_source_roots(source_dir: Path) -> Tuple[Path, Optional[Path], str]:
    source_dir = Path(source_dir)
    if (source_dir / "results" / "inference_cache" / "pgas").is_dir():
        results_root = source_dir / "results"
        cache_root = results_root / "inference_cache" / "pgas"
        output_root = results_root / "pgas_output"
        return cache_root, (output_root if output_root.is_dir() else None), "repo_root"

    if (source_dir / "inference_cache" / "pgas").is_dir():
        cache_root = source_dir / "inference_cache" / "pgas"
        output_root = source_dir / "pgas_output"
        return cache_root, (output_root if output_root.is_dir() else None), "run_root"

    if source_dir.name == "inference_cache" and (source_dir / "pgas").is_dir():
        cache_root = source_dir / "pgas"
        run_root = source_dir.parent
        output_root = run_root / "pgas_output"
        return cache_root, (output_root if output_root.is_dir() else None), "inference_cache"

    if source_dir.name == "pgas" and _looks_like_pgas_cache_root(source_dir):
        cache_root = source_dir
        run_root = source_dir.parent.parent if source_dir.parent.name == "inference_cache" else None
        output_root = (run_root / "pgas_output") if run_root is not None else None
        if output_root is not None and not output_root.is_dir():
            output_root = None
        return cache_root, output_root, "pgas_cache"

    if _looks_like_pgas_cache_root(source_dir):
        return source_dir, None, "pgas_cache"

    raise ValueError(
        "Could not resolve PGAS cache layout. Select a run root containing "
        "'inference_cache/pgas', an 'inference_cache' directory, a direct PGAS cache root, "
        "or a repository root containing 'results/inference_cache/pgas'."
    )


def _iter_cache_pairs(cache_root: Path) -> Iterable[Tuple[Path, Path, Path, Dict[str, object]]]:
    for dataset_dir in sorted(cache_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for meta_path in sorted(dataset_dir.glob("*.json")):
            mat_path = meta_path.with_suffix(".mat")
            if not mat_path.exists():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            yield dataset_dir, meta_path, mat_path, payload


def _candidate_output_roots(
    *,
    primary_output_root: Optional[Path],
    payload: Dict[str, object],
    source_dir: Path,
) -> List[Path]:
    out: List[Path] = []
    if primary_output_root is not None and primary_output_root.is_dir():
        out.append(primary_output_root)
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        raw = str(metadata.get("output_root", "")).strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = (source_dir / path).resolve()
            if path.is_dir():
                out.append(path)
    uniq: List[Path] = []
    seen: Set[str] = set()
    for root in out:
        token = str(root.resolve())
        if token in seen:
            continue
        seen.add(token)
        uniq.append(root)
    return uniq


def _extract_output_cache_tag(path: Path, *, prefix: str) -> Optional[str]:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".dat"):
        return None
    token = name[len(prefix):-4].strip()
    if not token:
        return None
    if "_trial" in token:
        token = token.split("_trial", 1)[0].strip()
    return token or None


def _parse_output_file(path: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    name = path.name
    patterns = [
        ("traj", "traj_samples_"),
        ("param", "param_samples_"),
        ("logp", "logp_"),
    ]
    for kind, prefix in patterns:
        if not name.startswith(prefix) or not name.endswith(".dat"):
            continue
        token = name[len(prefix):-4].strip()
        if not token:
            return None, None, None
        match = re.match(r"^(.*)_trial(\d+)$", token)
        if match:
            cache_tag = str(match.group(1)).strip()
            trial_idx = int(match.group(2))
        else:
            cache_tag = token
            trial_idx = None
        if not cache_tag:
            return None, None, None
        return kind, cache_tag, trial_idx
    return None, None, None


def _group_output_files_by_trial(files: List[Path]) -> Dict[int, List[Path]]:
    by_trial: Dict[int, Dict[str, List[Path]]] = {}
    for path in files:
        kind, _cache_tag, trial_idx = _parse_output_file(path)
        if kind is None:
            continue
        trial = 0 if trial_idx is None else int(trial_idx)
        row = by_trial.setdefault(trial, {"traj": [], "param": [], "logp": []})
        row[kind].append(path)
    out: Dict[int, List[Path]] = {}
    for trial_idx, row in sorted(by_trial.items()):
        traj = sorted(row["traj"], key=lambda p: p.name)
        params = sorted(row["param"], key=lambda p: p.name)
        logs = sorted(row["logp"], key=lambda p: p.name)
        if not traj or not params:
            continue
        out[int(trial_idx)] = traj + params + logs
    return out


def _rename_output_for_cache_tag(path: Path, *, cache_tag: str) -> str:
    kind, _src_tag, trial_idx = _parse_output_file(path)
    if kind == "traj":
        prefix = "traj_samples_"
    elif kind == "param":
        prefix = "param_samples_"
    elif kind == "logp":
        prefix = "logp_"
    else:
        return path.name
    if trial_idx is None:
        return f"{prefix}{cache_tag}.dat"
    return f"{prefix}{cache_tag}_trial{int(trial_idx)}.dat"


def _epoch_stems(valid_epoch_ids: Optional[Set[str]]) -> List[str]:
    if not valid_epoch_ids:
        return []
    stems: Set[str] = set()
    for epoch_id in valid_epoch_ids:
        token = str(epoch_id).strip()
        match = re.match(r"^(.*)_epoch(\d+)$", token)
        if match:
            stems.add(str(match.group(1)).strip())
    return sorted(stems, key=lambda s: len(s), reverse=True)


def _match_epoch_stem(cache_tag: str, stems: List[str]) -> Optional[str]:
    token = str(cache_tag).strip()
    for stem in stems:
        if token == stem or token.startswith(f"{stem}_"):
            return stem
    return None


def _cache_tag_with_epoch(cache_tag: str, stem: str, epoch_num: int) -> str:
    stem_epoch = f"{stem}_epoch{int(epoch_num)}"
    token = str(cache_tag).strip()
    if token == stem:
        return stem_epoch
    if token.startswith(f"{stem}_"):
        return f"{stem_epoch}{token[len(stem):]}"
    return stem_epoch


def _build_output_index(roots: List[Path]) -> Dict[str, Dict[str, List[Path]]]:
    index: Dict[str, Dict[str, List[Path]]] = {}
    patterns = [
        ("traj", "traj_samples_*.dat", "traj_samples_"),
        ("param", "param_samples_*.dat", "param_samples_"),
        ("logp", "logp_*.dat", "logp_"),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for kind, pattern, prefix in patterns:
            for path in root.rglob(pattern):
                cache_tag = _extract_output_cache_tag(path, prefix=prefix)
                if cache_tag is None:
                    continue
                bucket = index.setdefault(cache_tag, {"traj": [], "param": [], "logp": []})
                bucket[kind].append(path)

    for by_kind in index.values():
        for kind in ("traj", "param", "logp"):
            by_kind[kind] = sorted(by_kind[kind], key=lambda p: (str(p.parent), p.name))
    return index


def _best_output_parent(by_kind: Dict[str, List[Path]]) -> Optional[Path]:
    counts: Dict[Path, Dict[str, int]] = {}
    for kind in ("traj", "param", "logp"):
        for path in by_kind.get(kind, []):
            row = counts.setdefault(path.parent, {"traj": 0, "param": 0, "logp": 0})
            row[kind] += 1
    candidates: List[Tuple[int, int, str, Path]] = []
    for parent, row in counts.items():
        if row["traj"] <= 0 or row["param"] <= 0:
            continue
        total = row["traj"] + row["param"] + row["logp"]
        required = row["traj"] + row["param"]
        candidates.append((required, total, str(parent), parent))
    if not candidates:
        return None
    return sorted(candidates)[-1][3]


def _find_output_files(cache_tag: str, output_index: Dict[str, Dict[str, List[Path]]]) -> List[Path]:
    by_kind = output_index.get(cache_tag)
    if not by_kind:
        return []
    traj = list(by_kind.get("traj", []))
    params = list(by_kind.get("param", []))
    logs = list(by_kind.get("logp", []))
    if not traj or not params:
        return []
    parent = _best_output_parent(by_kind)
    if parent is not None:
        traj = [p for p in traj if p.parent == parent]
        params = [p for p in params if p.parent == parent]
        logs = [p for p in logs if p.parent == parent]
    if not traj or not params:
        return []
    files = traj + params + logs
    seen: Set[str] = set()
    out: List[Path] = []
    for path in files:
        token = str(path.resolve())
        if token in seen:
            continue
        seen.add(token)
        out.append(path)
    out.sort(key=lambda p: p.name)
    return out


def _root_key(roots: List[Path]) -> Tuple[str, ...]:
    keys: List[str] = []
    for root in roots:
        try:
            keys.append(str(root.resolve()))
        except Exception:
            keys.append(str(root))
    return tuple(keys)


def _output_index_for_roots(
    *,
    roots: List[Path],
    cache: Dict[Tuple[str, ...], Dict[str, Dict[str, List[Path]]]],
) -> Dict[str, Dict[str, List[Path]]]:
    key = _root_key(roots)
    if not key:
        return {}
    if key not in cache:
        cache[key] = _build_output_index(roots)
    return cache[key]


def _fallback_cache_tag_variants(cache_tag: str, epoch_id: str) -> List[str]:
    candidates: List[str] = []
    for token in (cache_tag, epoch_id):
        raw = str(token).strip()
        if not raw:
            continue
        candidates.append(raw)
        if "_epoch" in raw:
            stem = raw.split("_epoch", 1)[0]
            if stem:
                candidates.append(stem)
    out: List[str] = []
    seen: Set[str] = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _locate_output_files(
    *,
    cache_tag: str,
    epoch_id: str,
    output_index: Dict[str, Dict[str, List[Path]]],
) -> Tuple[List[Path], Optional[str]]:
    for candidate in _fallback_cache_tag_variants(cache_tag, epoch_id):
        files = _find_output_files(candidate, output_index)
        if files:
            return files, candidate
    return [], None


def import_pgas_cache_run(
    *,
    source_dir: Path,
    data_dir: Path,
    run_tag: str,
    valid_epoch_ids: Optional[Set[str]] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    source_dir = Path(source_dir).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not str(run_tag).strip():
        raise ValueError("run_tag must be non-empty.")
    valid_epoch_set: Optional[Set[str]] = None
    if valid_epoch_ids:
        valid_epoch_set = {str(e).strip() for e in valid_epoch_ids if str(e).strip()}
    epoch_stems = _epoch_stems(valid_epoch_set)

    cache_root, primary_output_root, source_kind = _resolve_source_roots(source_dir)
    dest_run_root = data_dir / "spike_inference" / str(run_tag).strip()
    dest_cache_root = dest_run_root / "inference_cache" / "pgas"
    dest_output_root = dest_run_root / "pgas_output"
    dest_temp_root = dest_run_root / "pgas_temp"

    dest_cache_root.mkdir(parents=True, exist_ok=True)
    dest_output_root.mkdir(parents=True, exist_ok=True)
    dest_temp_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "source_dir": str(source_dir),
        "source_kind": source_kind,
        "source_cache_root": str(cache_root),
        "source_output_root": str(primary_output_root) if primary_output_root is not None else None,
        "run_tag": str(run_tag).strip(),
        "dest_run_root": str(dest_run_root),
        "entries_scanned": 0,
        "entries_imported": 0,
        "entries_skipped_missing_epoch": 0,
        "entries_skipped_existing": 0,
        "entries_missing_outputs": 0,
        "entries_remapped_to_epoch": 0,
        "warnings": [],
        "imported_entries": [],
    }
    warnings = summary["warnings"]
    imported_entries = summary["imported_entries"]
    output_index_cache: Dict[Tuple[str, ...], Dict[str, Dict[str, List[Path]]]] = {}

    for dataset_dir, meta_path, mat_path, payload in _iter_cache_pairs(cache_root):
        summary["entries_scanned"] = int(summary["entries_scanned"]) + 1

        metadata = payload.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        cache_tag = str(metadata.get("cache_tag") or payload.get("dataset") or dataset_dir.name).strip()
        if not cache_tag:
            cache_tag = dataset_dir.name
        epoch_id = cache_tag_to_epoch_id(cache_tag)
        cache_key_src = str(payload.get("cache_key") or meta_path.stem).strip() or meta_path.stem

        output_roots = _candidate_output_roots(
            primary_output_root=primary_output_root,
            payload=payload,
            source_dir=source_dir,
        )
        output_index = _output_index_for_roots(roots=output_roots, cache=output_index_cache)
        output_files_raw, matched_output_tag = _locate_output_files(
            cache_tag=cache_tag,
            epoch_id=epoch_id,
            output_index=output_index,
        )
        output_by_trial = _group_output_files_by_trial(output_files_raw)

        targets: List[Dict[str, object]] = []
        if not valid_epoch_set or epoch_id in valid_epoch_set:
            targets.append(
                {
                    "epoch_id": epoch_id,
                    "cache_tag": cache_tag,
                    "cache_key": cache_key_src,
                    "output_files": output_files_raw,
                    "trial_idx": None,
                    "remapped": False,
                }
            )
        else:
            stem = _match_epoch_stem(cache_tag, epoch_stems)
            if stem and output_by_trial:
                for trial_idx, files_for_trial in sorted(output_by_trial.items()):
                    epoch_num = int(trial_idx) + 1
                    mapped_epoch_id = f"{stem}_epoch{epoch_num}"
                    if valid_epoch_set and mapped_epoch_id not in valid_epoch_set:
                        continue
                    mapped_tag = _cache_tag_with_epoch(cache_tag, stem, epoch_num)
                    mapped_key = f"{cache_key_src}_trial{int(trial_idx)}"
                    targets.append(
                        {
                            "epoch_id": mapped_epoch_id,
                            "cache_tag": mapped_tag,
                            "cache_key": mapped_key,
                            "output_files": files_for_trial,
                            "trial_idx": int(trial_idx),
                            "remapped": True,
                        }
                    )
                if targets:
                    summary["entries_remapped_to_epoch"] = int(summary["entries_remapped_to_epoch"]) + 1
            if not targets and stem:
                default_epoch = f"{stem}_epoch1"
                if not valid_epoch_set or default_epoch in valid_epoch_set:
                    mapped_tag = _cache_tag_with_epoch(cache_tag, stem, 1)
                    targets.append(
                        {
                            "epoch_id": default_epoch,
                            "cache_tag": mapped_tag,
                            "cache_key": f"{cache_key_src}_trial0",
                            "output_files": output_by_trial.get(0, []),
                            "trial_idx": 0,
                            "remapped": True,
                        }
                    )
                    summary["entries_remapped_to_epoch"] = int(summary["entries_remapped_to_epoch"]) + 1
                    warnings.append(
                        f"Remapped {cache_tag} to {default_epoch} using default trial0 (no explicit trial mapping found)."
                    )

        if not targets:
            summary["entries_skipped_missing_epoch"] = int(summary["entries_skipped_missing_epoch"]) + 1
            warnings.append(
                f"Skipping {cache_tag}: resolved epoch_id '{epoch_id}' not found in selected dataset."
            )
            continue
        for target in targets:
            target_epoch_id = str(target["epoch_id"])
            target_cache_tag = str(target["cache_tag"])
            target_cache_key = str(target["cache_key"])
            target_files = list(target.get("output_files", []))
            trial_idx = target.get("trial_idx")
            remapped = bool(target.get("remapped", False))

            dest_dataset_dir = dest_cache_root / dataset_dir.name
            dest_dataset_dir.mkdir(parents=True, exist_ok=True)
            dest_meta = dest_dataset_dir / f"{target_cache_key}.json"
            dest_mat = dest_dataset_dir / f"{target_cache_key}.mat"

            if not overwrite and (dest_meta.exists() or dest_mat.exists()):
                summary["entries_skipped_existing"] = int(summary["entries_skipped_existing"]) + 1
                warnings.append(
                    f"Skipping {target_cache_tag}: destination cache files already exist for key '{target_cache_key}'."
                )
                continue

            if overwrite or not dest_mat.exists():
                shutil.copy2(mat_path, dest_mat)

            payload_out = dict(payload)
            md = payload_out.get("metadata", {})
            if not isinstance(md, dict):
                md = {}
            md["output_root"] = str(dest_output_root)
            md["cache_tag"] = target_cache_tag
            payload_out["metadata"] = md
            payload_out["dataset"] = target_cache_tag
            payload_out["method"] = "pgas"
            payload_out["cache_key"] = target_cache_key
            dest_meta.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")

            copied_outputs = 0
            if target_files:
                for src in target_files:
                    dest_name = _rename_output_for_cache_tag(src, cache_tag=target_cache_tag)
                    dest = dest_output_root / dest_name
                    if overwrite or not dest.exists():
                        shutil.copy2(src, dest)
                    copied_outputs += 1
                if matched_output_tag and matched_output_tag != cache_tag:
                    warnings.append(
                        f"Imported cache {cache_tag}: matched output files using fallback tag '{matched_output_tag}'."
                    )
            else:
                summary["entries_missing_outputs"] = int(summary["entries_missing_outputs"]) + 1
                warnings.append(
                    f"Imported cache {target_cache_tag} but could not find matching traj/param files in source pgas_output."
                )

            if remapped:
                warnings.append(
                    f"Remapped source cache {cache_tag} trial{int(trial_idx) if trial_idx is not None else 0} -> {target_cache_tag}."
                )

            summary["entries_imported"] = int(summary["entries_imported"]) + 1
            imported_entries.append(
                {
                    "cache_tag": target_cache_tag,
                    "epoch_id": target_epoch_id,
                    "cache_key": target_cache_key,
                    "source_cache_tag": cache_tag,
                    "source_trial_idx": trial_idx,
                    "source_meta": str(meta_path),
                    "source_mat": str(mat_path),
                    "copied_output_files": copied_outputs,
                }
            )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    manifest_path = dest_run_root / "imported_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary["manifest_path"] = str(manifest_path)
    return summary
