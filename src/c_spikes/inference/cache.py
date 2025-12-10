from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import scipy.io as sio

from .types import MethodResult, compute_config_signature, ensure_serializable


CACHE_ROOT = Path("results") / "inference_cache"


def get_cache_paths(
    method: str,
    dataset_tag: str,
    config_hash: str,
    *,
    cache_root: Path = CACHE_ROOT,
) -> Tuple[Path, Path]:
    cache_dir = cache_root / method / dataset_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{config_hash}.mat", cache_dir / f"{config_hash}.json"


def save_method_cache(
    method: str,
    dataset_tag: str,
    result: MethodResult,
    config: Mapping[str, Any],
    trace_hash: str,
    *,
    cache_root: Path = CACHE_ROOT,
) -> None:
    config_hash, config_ser = compute_config_signature(dict(config))
    mat_path, meta_path = get_cache_paths(method, dataset_tag, config_hash, cache_root=cache_root)
    payload = {
        "time_stamps": np.asarray(result.time_stamps),
        "spike_prob": np.asarray(result.spike_prob),
    }
    if result.reconstruction is not None:
        payload["reconstruction"] = np.asarray(result.reconstruction)
    if result.discrete_spikes is not None:
        payload["discrete_spikes"] = np.asarray(result.discrete_spikes)
    sio.savemat(mat_path, payload, do_compression=True)
    meta = {
        "dataset": dataset_tag,
        "method": method,
        "config": config_ser,
        "trace_hash": trace_hash,
        "sampling_rate": float(result.sampling_rate),
        "metadata": ensure_serializable(result.metadata),
        "cache_key": config_hash,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_method_cache(
    method: str,
    dataset_tag: str,
    config: Mapping[str, Any],
    trace_hash: str,
    *,
    cache_root: Path = CACHE_ROOT,
    allow_mismatched_trace: bool = False,
) -> Optional[MethodResult]:
    config_hash, _ = compute_config_signature(dict(config))
    mat_path, meta_path = get_cache_paths(method, dataset_tag, config_hash, cache_root=cache_root)
    candidates: list[Tuple[Path, Path]] = []
    if mat_path.exists() and meta_path.exists():
        candidates.append((mat_path, meta_path))
    cache_dir = meta_path.parent
    if not candidates and cache_dir.exists():
        for meta_candidate in sorted(cache_dir.glob("*.json")):
            mat_candidate = meta_candidate.with_suffix(".mat")
            if not mat_candidate.exists():
                continue
            candidates.append((mat_candidate, meta_candidate))

    for mat_candidate, meta_candidate in candidates:
        try:
            with meta_candidate.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not allow_mismatched_trace and meta.get("trace_hash") != trace_hash:
            continue
        if meta.get("dataset") not in {dataset_tag, meta.get("metadata", {}).get("cache_tag")}:
            continue
        data = sio.loadmat(mat_candidate)
        time_stamps = np.asarray(data.get("time_stamps")).squeeze()
        spike_prob = np.asarray(data.get("spike_prob")).squeeze()
        reconstruction = data.get("reconstruction")
        reconstruction = None if reconstruction is None else np.asarray(reconstruction).squeeze()
        discrete = data.get("discrete_spikes")
        discrete = None if discrete is None else np.asarray(discrete).squeeze()
        return MethodResult(
            name=method,
            time_stamps=time_stamps,
            spike_prob=spike_prob,
            sampling_rate=float(meta.get("sampling_rate", 0.0)),
            metadata=meta.get("metadata", {}),
            reconstruction=reconstruction,
            discrete_spikes=discrete,
        )
    return None


