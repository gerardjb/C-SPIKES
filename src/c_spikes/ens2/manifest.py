from __future__ import annotations

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


__all__ = ["add_synthetic_entry", "add_training_entry"]
