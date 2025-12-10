from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from c_spikes.model_eval.model_eval import smooth_prediction, smooth_spike_train

from .types import MethodResult


def build_ground_truth_series(
    spike_times: np.ndarray,
    global_start: float,
    global_end: float,
    reference_fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    spikes = np.asarray(spike_times, dtype=float).ravel()
    duration = float(global_end - global_start)
    n_samples = max(2, int(np.round(duration * reference_fs)) + 1)
    time_grid = np.linspace(global_start, global_end, n_samples, dtype=np.float64)
    series = np.zeros_like(time_grid)
    if spikes.size == 0:
        return time_grid, series
    bin_idx = np.searchsorted(time_grid, spikes, side="left")
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < time_grid.size)]
    if bin_idx.size:
        np.add.at(series, bin_idx, 1.0)
    smoothed = smooth_spike_train(series, reference_fs, sigma_ms=50.0)
    return time_grid, smoothed


def segment_indices(times: np.ndarray, fs_est: float, gap_factor: float = 4.0) -> List[slice]:
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return []
    diffs = np.diff(times)
    if not np.isfinite(fs_est) or fs_est <= 0:
        raise ValueError("fs_est must be positive and finite.")
    base_dt = 1.0 / fs_est
    threshold = gap_factor * base_dt
    breaks = np.where((diffs > threshold) | ~np.isfinite(diffs))[0] + 1
    indices = np.concatenate([[0], breaks, [times.size]])
    segments: List[slice] = []
    for start, end in zip(indices[:-1], indices[1:]):
        if end > start:
            segments.append(slice(start, end))
    return segments


def resample_prediction_to_reference(
    times: np.ndarray,
    values: np.ndarray,
    reference_time: np.ndarray,
    *,
    fs_est: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    reference_time = np.asarray(reference_time, dtype=np.float64)
    result = np.full(reference_time.shape, np.nan, dtype=np.float64)
    segments = segment_indices(times, fs_est)
    for seg in segments:
        seg_times = times[seg]
        seg_values = values[seg]
        if seg_times.size == 0:
            continue
        mask = (reference_time >= seg_times[0]) & (reference_time <= seg_times[-1])
        if not mask.any():
            continue
        if seg_times.size == 1:
            idx = np.argmin(np.abs(reference_time - seg_times[0]))
            result[idx] = seg_values[0]
            continue
        result[mask] = np.interp(reference_time[mask], seg_times, seg_values)
    return result


def compute_correlations(
    methods: Sequence[MethodResult],
    reference_time: np.ndarray,
    reference_trace: np.ndarray,
    *,
    sigma_ms: float = 50.0,
    windows: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    correlations: Dict[str, float] = {}
    if windows:
        window_mask = np.zeros(reference_time.shape, dtype=bool)
        for start, end in windows:
            if not np.isfinite(start) or not np.isfinite(end) or end < start:
                continue
            window_mask |= (reference_time >= start) & (reference_time <= end)
    else:
        window_mask = np.ones(reference_time.shape, dtype=bool)

    for method in methods:
        pred_smoothed = smooth_prediction(method.spike_prob, method.sampling_rate, sigma_ms=sigma_ms)
        aligned = resample_prediction_to_reference(
            method.time_stamps,
            pred_smoothed,
            reference_time,
            fs_est=method.sampling_rate,
        )
        valid = np.isfinite(aligned) & np.isfinite(reference_trace) & window_mask
        if valid.sum() < 2:
            correlations[method.name] = float("nan")
            continue
        x = reference_trace[valid]
        y = aligned[valid]
        x -= np.mean(x)
        y -= np.mean(y)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        correlations[method.name] = float(np.nan if denom == 0 else np.dot(x, y) / denom)
    return correlations


