from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrialSeries:
    times: np.ndarray
    values: np.ndarray

    def current_fs(self) -> float:
        diffs = np.diff(self.times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            raise ValueError("Insufficient samples to compute sampling rate.")
        return float(1.0 / np.median(diffs))


@dataclass
class MethodResult:
    name: str
    time_stamps: np.ndarray
    spike_prob: np.ndarray
    sampling_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Optional[np.ndarray] = None
    discrete_spikes: Optional[np.ndarray] = None


def ensure_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [ensure_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    return obj


def compute_config_signature(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    config_ser = ensure_serializable(config)
    import json

    encoded = json.dumps(config_ser, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()[:16], config_ser


def hash_series(times: np.ndarray, values: np.ndarray) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(np.asarray(times, dtype=np.float64).tobytes())
    h.update(np.asarray(values, dtype=np.float32).tobytes())
    return h.hexdigest()


def hash_array(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(arr).astype(np.float32).tobytes()).hexdigest()


def flatten_trials(trials: Sequence[TrialSeries]) -> Tuple[np.ndarray, np.ndarray]:
    times = np.concatenate([trial.times for trial in trials])
    values = np.concatenate([trial.values for trial in trials])
    order = np.argsort(times)
    return times[order], values[order]


def compute_sampling_rate(times: np.ndarray) -> float:
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("Insufficient samples to compute sampling rate.")
    return float(1.0 / np.median(diffs))


def extract_spike_times(result: MethodResult) -> Optional[np.ndarray]:
    if result.discrete_spikes is None:
        return None
    counts = np.asarray(result.discrete_spikes, dtype=float)
    times: List[float] = []
    for t, c in zip(result.time_stamps, counts):
        if not np.isfinite(c) or not np.isfinite(t):
            continue
        n = int(round(c))
        if n > 0:
            times.extend([float(t)] * n)
    return np.asarray(times, dtype=float)

