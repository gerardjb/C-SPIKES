from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from .types import MethodResult, compute_config_signature, ensure_serializable


CACHE_ROOT = Path("results") / "inference_cache"


def get_cache_root() -> Path:
    return CACHE_ROOT


def set_cache_root(root: Path) -> None:
    global CACHE_ROOT
    CACHE_ROOT = Path(root)


def _resolve_cache_root(cache_root: Optional[Path]) -> Path:
    return CACHE_ROOT if cache_root is None else Path(cache_root)


def get_cache_paths(
    method: str,
    dataset_tag: str,
    config_hash: str,
    *,
    cache_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    cache_dir = _resolve_cache_root(cache_root) / method / dataset_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{config_hash}.mat", cache_dir / f"{config_hash}.json"


def _load_method_cache_result(method: str, mat_path: Path, meta: Mapping[str, Any]) -> MethodResult:
    data = sio.loadmat(mat_path)
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
        metadata=dict(meta.get("metadata", {})),
        reconstruction=reconstruction,
        discrete_spikes=discrete,
    )


def save_method_cache(
    method: str,
    dataset_tag: str,
    result: MethodResult,
    config: Mapping[str, Any],
    trace_hash: str,
    *,
    cache_root: Optional[Path] = None,
    extra_payload: Optional[Mapping[str, Any]] = None,
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
    if extra_payload:
        payload.update(dict(extra_payload))
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
    cache_root: Optional[Path] = None,
    allow_mismatched_trace: bool = False,
) -> Optional[MethodResult]:
    config_hash, config_ser = compute_config_signature(dict(config))
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
        # Never cross-load caches across different configs (e.g., different pretrained_dir).
        # This is critical for sweeps where the trace_hash/dataset_tag match but the model changes.
        cache_key = meta.get("cache_key")
        if cache_key is not None:
            if cache_key != config_hash:
                continue
        else:
            if meta.get("config") != config_ser:
                continue
        if not allow_mismatched_trace and meta.get("trace_hash") != trace_hash:
            continue
        if meta.get("dataset") not in {dataset_tag, meta.get("metadata", {}).get("cache_tag")}:
            continue
        return _load_method_cache_result(method, mat_candidate, meta)
    return None


def _stable_config_matches(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    keys: Sequence[str],
) -> bool:
    for key in keys:
        if key not in expected:
            continue
        if key not in actual:
            return False
        expected_value = expected[key]
        actual_value = actual[key]
        if isinstance(expected_value, float) or isinstance(actual_value, float):
            try:
                if not np.isclose(float(actual_value), float(expected_value), rtol=1e-7, atol=1e-12):
                    return False
                continue
            except (TypeError, ValueError):
                return False
        if actual_value != expected_value:
            return False
    return True


def load_method_cache_legacy_compatible(
    method: str,
    dataset_tags: Iterable[str],
    config: Mapping[str, Any],
    trace_hash: str,
    *,
    stable_config_keys: Sequence[str],
    cache_root: Optional[Path] = None,
) -> Optional[MethodResult]:
    """Load legacy caches whose absolute path config fields differ across worktrees.

    This intentionally remains stricter than a blind tag lookup: the cache tag must
    match one of ``dataset_tags``, the trace hash must match, and selected stable
    config fields must match. It exists for older PGAS caches whose config hashes
    included absolute constants/gparam paths from another worktree/build.
    """

    root = _resolve_cache_root(cache_root) / method
    for dataset_tag in dataset_tags:
        cache_dir = root / str(dataset_tag)
        if not cache_dir.exists():
            continue
        for meta_path in sorted(cache_dir.glob("*.json")):
            mat_path = meta_path.with_suffix(".mat")
            if not mat_path.exists():
                continue
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("trace_hash") != trace_hash:
                continue
            if meta.get("dataset") not in {dataset_tag, meta.get("metadata", {}).get("cache_tag")}:
                continue
            actual_config = meta.get("config", {})
            if not isinstance(actual_config, Mapping):
                continue
            if not _stable_config_matches(actual_config, config, stable_config_keys):
                continue
            result = _load_method_cache_result(method, mat_path, meta)
            result.metadata.setdefault("cache_tag", dataset_tag)
            result.metadata.setdefault("cache_style", "legacy_path_compatible")
            result.metadata.setdefault("cache_key", meta.get("cache_key"))
            return result
    return None
