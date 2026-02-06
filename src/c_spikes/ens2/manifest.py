from __future__ import annotations

import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _now_iso() -> str:
    return datetime.now().isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_dir_quick(dir_path: Path) -> str:
    """
    Lightweight directory fingerprint: hashes filenames + sizes + mtimes.
    Avoids reading full .mat contents to stay fast.
    """
    h = hashlib.sha256()
    for file in sorted(dir_path.glob("*.mat")):
        st = file.stat()
        h.update(file.name.encode("utf-8"))
        h.update(str(st.st_size).encode("ascii"))
        h.update(str(int(st.st_mtime)).encode("ascii"))
    return h.hexdigest()


def _hash_array(arr: np.ndarray) -> str:
    from c_spikes.inference.types import hash_array  # reuse existing helper

    return hash_array(np.asarray(arr, dtype=np.float32))


def _ensure_manifest(path: Path, model_name: Optional[str] = None) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    else:
        manifest = {
            "model_name": model_name or "",
            "created": _now_iso(),
            "updated": _now_iso(),
            "synthetic_entries": [],
            "training": {},
        }
    if model_name and not manifest.get("model_name"):
        manifest["model_name"] = model_name
    return manifest


def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def add_synthetic_entry(
    manifest_path: Path,
    *,
    model_name: Optional[str],
    param_samples_path: Path,
    gparam_path: Path,
    burnin: int,
    cparams: np.ndarray,
    syn_gen_params: Dict[str, Any],
    output_dir: Path,
    run_tag: Optional[str],
) -> Dict[str, Any]:
    manifest = _ensure_manifest(manifest_path, model_name=model_name)
    entry = {
        "timestamp": _now_iso(),
        "run_tag": run_tag,
        "param_samples_path": str(param_samples_path),
        "param_samples_sha256": _sha256_file(param_samples_path),
        "gparam_path": str(gparam_path),
        "gparam_sha256": _sha256_file(gparam_path),
        "burnin": int(burnin),
        "cparams_mean": np.asarray(cparams, dtype=float).tolist(),
        "cparams_hash": _hash_array(cparams),
        "syn_gen": syn_gen_params,
        "synth_dir_hash": _hash_dir_quick(output_dir),
    }
    manifest.setdefault("synthetic_entries", []).append(entry)
    manifest["updated"] = _now_iso()
    _save_manifest(manifest_path, manifest)
    return manifest


def add_training_entry(
    manifest_path: Path,
    *,
    model_name: str,
    checkpoint_path: Path,
    neuron_type: str,
    sampling_rate: float,
    smoothing_std: float,
    run_tag: Optional[str],
    ground_truth_dir: Optional[Path | Sequence[Path | str]],
    device: Optional[str],
) -> Dict[str, Any]:
    manifest = _ensure_manifest(manifest_path, model_name=model_name)
    if ground_truth_dir is None:
        gt_dir_val: Optional[Any] = None
    elif isinstance(ground_truth_dir, (str, Path)):
        gt_dir_val = str(ground_truth_dir)
    else:
        gt_dir_val = [str(p) for p in ground_truth_dir]
    training_entry = {
        "timestamp": _now_iso(),
        "run_tag": run_tag,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "neuron_type": neuron_type,
        "sampling_rate": float(sampling_rate),
        "smoothing_std": float(smoothing_std),
        "ground_truth_dir": gt_dir_val,
        "device": device,
    }
    manifest["training"] = training_entry
    manifest["updated"] = _now_iso()
    _save_manifest(manifest_path, manifest)
    return manifest


def resolve_model_dir_by_run_tag(run_tag: str, model_roots: Sequence[Path]) -> Path:
    """
    Resolve an ENS2 model directory by matching `run_tag` in an `ens2_manifest.json`.

    Matches either:
      - manifest["training"]["run_tag"]
      - any entry["run_tag"] in manifest["synthetic_entries"]

    Notes:
      - We intentionally skip descending into nested "results/" directories to avoid
        picking up mirrored training artifacts under model folders.
      - If multiple matches are found, we raise an error and ask the caller to disambiguate
        by passing an explicit `--ens2-pretrained-root` (or narrower model_roots).
    """
    tag = str(run_tag).strip()
    if not tag:
        raise ValueError("run_tag must be a non-empty string")

    roots = [Path(p).expanduser() for p in model_roots]
    matches: list[Path] = []

    def _has_checkpoint(model_dir: Path) -> bool:
        return (model_dir / "exc_ens2_pub.pt").exists() or (model_dir / "inh_ens2_pub.pt").exists()

    for root in roots:
        if not root.exists():
            continue
        # Convenience: allow the run_tag to be a direct model directory name.
        direct = root / tag
        if direct.is_dir() and _has_checkpoint(direct):
            return direct.resolve()

        for dirpath, dirnames, filenames in os.walk(root):
            # Avoid traversing nested model outputs (common in training artifacts).
            dirnames[:] = [
                d
                for d in dirnames
                if d not in {"results", "__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
            ]
            if "ens2_manifest.json" not in filenames:
                continue
            manifest_path = Path(dirpath) / "ens2_manifest.json"
            model_dir = manifest_path.parent
            if not _has_checkpoint(model_dir):
                continue
            try:
                with manifest_path.open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue

            if manifest.get("training", {}).get("run_tag") == tag:
                matches.append(model_dir.resolve())
                continue
            for entry in manifest.get("synthetic_entries", []) or []:
                if isinstance(entry, dict) and entry.get("run_tag") == tag:
                    matches.append(model_dir.resolve())
                    break

    # De-dupe while preserving order.
    uniq: list[Path] = []
    seen: set[str] = set()
    for m in matches:
        key = str(m)
        if key not in seen:
            uniq.append(m)
            seen.add(key)

    if not uniq:
        roots_str = ", ".join(str(r) for r in roots)
        raise FileNotFoundError(f"No ENS2 model found for run_tag='{tag}' under: {roots_str}")
    if len(uniq) > 1:
        lines = "\n".join(f"  - {p}" for p in uniq)
        raise RuntimeError(f"Multiple ENS2 models matched run_tag='{tag}':\n{lines}")
    return uniq[0]


__all__ = ["add_synthetic_entry", "add_training_entry", "resolve_model_dir_by_run_tag"]
