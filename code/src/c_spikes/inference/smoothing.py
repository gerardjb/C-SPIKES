from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import TrialSeries


SMOOTHING_LEVELS: Sequence[Tuple[str, Optional[float]]] = [
    ("raw", None),
    ("30Hz", 30.0),
    ("10Hz", 10.0),
]


def resolve_smoothing_levels(
    selection: Optional[Sequence[str]],
) -> Sequence[Tuple[str, Optional[float]]]:
    if not selection:
        return SMOOTHING_LEVELS
    mapping: Dict[str, Tuple[str, Optional[float]]] = {
        "raw": ("raw", None),
        "30": ("30Hz", 30.0),
        "30hz": ("30Hz", 30.0),
        "10": ("10Hz", 10.0),
        "10hz": ("10Hz", 10.0),
    }
    resolved: List[Tuple[str, Optional[float]]] = []
    for token in selection:
        key = token.strip().lower()
        if key not in mapping:
            raise ValueError(
                f"Invalid smoothing level '{token}'. Expected one of: "
                + ", ".join(sorted(mapping.keys()))
            )
        entry = mapping[key]
        if entry not in resolved:
            resolved.append(entry)
    return resolved


def mean_downsample_trace(times: np.ndarray, values: np.ndarray, target_fs: float) -> TrialSeries:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size == 0:
        return TrialSeries(times=times, values=values)
    dt = np.median(np.diff(times))
    fs = 1.0 / dt
    if fs <= target_fs:
        return TrialSeries(times=times.copy(), values=values.copy())
    ratio = fs / target_fs
    B = int(round(ratio))
    if not np.isclose(ratio, B, atol=1e-6):
        duration = times[-1] - times[0]
        n_target = int(np.round(duration * target_fs)) + 1
        new_times = np.linspace(times[0], times[-1], n_target)
        new_values = np.interp(new_times, times, values)
        return TrialSeries(times=new_times, values=new_values)
    n_trim = (values.size // B) * B
    if n_trim == 0:
        return TrialSeries(times=times[:1], values=values[:1])
    values_trim = values[:n_trim].reshape(-1, B)
    times_trim = times[:n_trim].reshape(-1, B)
    values_ds = values_trim.mean(axis=1)
    times_ds = times_trim.mean(axis=1)
    return TrialSeries(times=times_ds, values=values_ds)


def resample_trial_to_fs(trial: TrialSeries, target_fs: float, tol: float = 1e-3) -> TrialSeries:
    if target_fs <= 0:
        raise ValueError("target_fs must be positive")
    times = np.asarray(trial.times, dtype=np.float64)
    values = np.asarray(trial.values, dtype=np.float64)
    if times.size <= 1:
        return TrialSeries(times=times.copy(), values=values.copy())

    dt = np.median(np.diff(times))
    current_fs = 1.0 / dt
    if np.isclose(current_fs, target_fs, rtol=tol, atol=tol * target_fs):
        return TrialSeries(times=times.copy(), values=values.copy())

    if current_fs > target_fs:
        return mean_downsample_trace(times, values, target_fs)

    duration = float(times[-1] - times[0])
    if duration <= 0:
        return TrialSeries(times=times.copy(), values=values.copy())
    n_target = max(2, int(np.round(duration * target_fs)) + 1)
    new_times = np.linspace(times[0], times[-1], n_target, dtype=np.float64)
    new_values = np.interp(new_times, times, values)
    return TrialSeries(times=new_times, values=new_values)


def resample_trials_to_fs(trials: Sequence[TrialSeries], target_fs: float) -> List[TrialSeries]:
    return [resample_trial_to_fs(trial, target_fs) for trial in trials]


