from __future__ import annotations

import json
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
        "'inference_cache/pgas', an 'inference_cache' directory, or a direct PGAS cache root."
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
) -> List[Path]:
    out: List[Path] = []
    if primary_output_root is not None and primary_output_root.is_dir():
        out.append(primary_output_root)
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        raw = str(metadata.get("output_root", "")).strip()
        if raw:
            path = Path(raw)
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


def _find_output_files(cache_tag: str, roots: List[Path]) -> List[Path]:
    for root in roots:
        traj = sorted(root.glob(f"traj_samples_{cache_tag}_trial*.dat"))
        params = sorted(root.glob(f"param_samples_{cache_tag}_trial*.dat"))
        logs = sorted(root.glob(f"logp_{cache_tag}_trial*.dat"))
        if traj and params:
            return traj + params + logs
    return []


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
        "warnings": [],
        "imported_entries": [],
    }
    warnings = summary["warnings"]
    imported_entries = summary["imported_entries"]

    for dataset_dir, meta_path, mat_path, payload in _iter_cache_pairs(cache_root):
        summary["entries_scanned"] = int(summary["entries_scanned"]) + 1

        metadata = payload.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        cache_tag = str(metadata.get("cache_tag") or payload.get("dataset") or dataset_dir.name).strip()
        if not cache_tag:
            cache_tag = dataset_dir.name
        epoch_id = cache_tag_to_epoch_id(cache_tag)

        if valid_epoch_ids and epoch_id not in valid_epoch_ids:
            summary["entries_skipped_missing_epoch"] = int(summary["entries_skipped_missing_epoch"]) + 1
            warnings.append(
                f"Skipping {cache_tag}: resolved epoch_id '{epoch_id}' not found in selected dataset."
            )
            continue

        cache_key = str(payload.get("cache_key") or meta_path.stem).strip() or meta_path.stem
        dest_dataset_dir = dest_cache_root / dataset_dir.name
        dest_dataset_dir.mkdir(parents=True, exist_ok=True)
        dest_meta = dest_dataset_dir / f"{cache_key}.json"
        dest_mat = dest_dataset_dir / f"{cache_key}.mat"

        if not overwrite and (dest_meta.exists() or dest_mat.exists()):
            summary["entries_skipped_existing"] = int(summary["entries_skipped_existing"]) + 1
            warnings.append(
                f"Skipping {cache_tag}: destination cache files already exist for key '{cache_key}'."
            )
            continue

        if overwrite or not dest_mat.exists():
            shutil.copy2(mat_path, dest_mat)

        payload_out = dict(payload)
        md = payload_out.get("metadata", {})
        if not isinstance(md, dict):
            md = {}
        md["output_root"] = str(dest_output_root)
        md.setdefault("cache_tag", cache_tag)
        payload_out["metadata"] = md
        payload_out.setdefault("dataset", cache_tag)
        payload_out.setdefault("method", "pgas")
        payload_out.setdefault("cache_key", cache_key)
        dest_meta.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")

        output_roots = _candidate_output_roots(primary_output_root=primary_output_root, payload=payload)
        output_files = _find_output_files(cache_tag, output_roots)
        copied_outputs = 0
        if output_files:
            for src in output_files:
                dest = dest_output_root / src.name
                if overwrite or not dest.exists():
                    shutil.copy2(src, dest)
                copied_outputs += 1
        else:
            summary["entries_missing_outputs"] = int(summary["entries_missing_outputs"]) + 1
            warnings.append(
                f"Imported cache {cache_tag} but could not find matching traj/param files in source pgas_output."
            )

        summary["entries_imported"] = int(summary["entries_imported"]) + 1
        imported_entries.append(
            {
                "cache_tag": cache_tag,
                "epoch_id": epoch_id,
                "cache_key": cache_key,
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

