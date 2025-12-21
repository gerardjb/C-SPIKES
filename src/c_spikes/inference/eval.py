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
    *,
    sigma_ms: float = 50.0,
    binning: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    spikes = np.asarray(spike_times, dtype=float).ravel()
    duration = float(global_end - global_start)
    if duration <= 0:
        raise ValueError("global_end must exceed global_start when building ground truth.")

    # Match the length convention used by smooth_spike_train (ceil instead of round).
    n_samples = int(np.ceil(duration * reference_fs)) + 1
    time_grid = np.linspace(global_start, global_end, n_samples, dtype=np.float64)
    if spikes.size == 0:
        return time_grid, np.zeros_like(time_grid)

    valid_spikes = spikes[np.isfinite(spikes)]
    valid_spikes = valid_spikes[(valid_spikes >= global_start) & (valid_spikes <= global_end)]
    rel_spikes = valid_spikes - float(global_start)

    smoothed = smooth_spike_train(
        rel_spikes,
        reference_fs,
        duration=duration,
        sigma_ms=sigma_ms,
        binning=binning,
    )
    # smooth_spike_train derives length from the provided duration; if rounding differs,
    # interpolate to align with the reference grid.
    if smoothed.size != time_grid.size:
        smoothed_time = np.linspace(global_start, global_end, smoothed.size, dtype=np.float64)
        smoothed = np.interp(time_grid, smoothed_time, smoothed)
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
    reference_time = np.asarray(reference_time, dtype=float).ravel()
    reference_trace = np.asarray(reference_trace, dtype=float).ravel()
    if reference_time.shape != reference_trace.shape:
        raise ValueError(
            f"reference_time shape {reference_time.shape} does not match reference_trace {reference_trace.shape}"
        )

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


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask] - float(np.mean(x[mask]))
    y = y[mask] - float(np.mean(y[mask]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(x, y) / denom)


def _preprocess_prediction_for_correlation(method: str, values: np.ndarray) -> np.ndarray:
    """
    Method-specific fixes needed for fair GT correlation comparisons.

    PGAS `spike_prob` behaves like per-bin spike counts reported on the *right edge* of each
    interval; shifting left by one sample aligns peaks with the start-of-bin GT convention.
    """
    y = np.asarray(values, dtype=np.float64).ravel()
    if method == "pgas" and y.size >= 2:
        return np.concatenate([y[1:], np.array([0.0], dtype=y.dtype)])
    return y


def _select_segment_for_window(
    segments: Sequence[slice],
    times: np.ndarray,
    window_idx: int,
    start: float,
    end: float,
    *,
    n_windows: Optional[int] = None,
) -> Optional[slice]:
    if not segments:
        return None
    if n_windows is not None and len(segments) == int(n_windows) and 0 <= window_idx < len(segments):
        return segments[window_idx]

    times = np.asarray(times, dtype=np.float64).ravel()
    best: Optional[slice] = None
    best_overlap = -1.0
    for seg in segments:
        seg_times = times[seg]
        if seg_times.size == 0:
            continue
        seg_start = float(seg_times[0])
        seg_end = float(seg_times[-1])
        overlap = max(0.0, min(seg_end, float(end)) - max(seg_start, float(start)))
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg
    if best_overlap <= 0.0:
        return None
    return best


def compute_correlations_windowed(
    methods: Sequence[MethodResult],
    spike_times: np.ndarray,
    windows: Sequence[Tuple[float, float]],
    *,
    reference_fs: float,
    sigma_ms: float = 50.0,
    binning: str = "linear",
) -> Dict[str, float]:
    """
    Compute correlations over a set of windows using per-window (start/end anchored) reference grids.

    This avoids the subtle time-grid offset introduced by building a single global linspace and then
    masking to windows, which can materially change correlations at low reference_fs / small sigma.
    """
    if not windows:
        return {m.name: float("nan") for m in methods}

    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    segments_by_method: Dict[str, List[slice]] = {}
    for method in methods:
        segments_by_method[method.name] = segment_indices(method.time_stamps, method.sampling_rate)

    ref_concat: List[np.ndarray] = []
    pred_concat: Dict[str, List[np.ndarray]] = {m.name: [] for m in methods}
    for idx, (start, end) in enumerate(windows):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        ref_time, ref_trace = build_ground_truth_series(
            spike_times,
            float(start),
            float(end),
            reference_fs=float(reference_fs),
            sigma_ms=float(sigma_ms),
            binning=binning,
        )
        ref_concat.append(np.asarray(ref_trace, dtype=np.float64).ravel())
        for method in methods:
            seg = _select_segment_for_window(
                segments_by_method.get(method.name, []),
                method.time_stamps,
                idx,
                float(start),
                float(end),
                n_windows=len(windows),
            )
            if seg is None:
                pred_concat[method.name].append(np.full(ref_trace.shape, np.nan, dtype=np.float64))
                continue
            t_seg = np.asarray(method.time_stamps, dtype=np.float64).ravel()[seg]
            y_seg = np.asarray(method.spike_prob, dtype=np.float64).ravel()[seg]
            mm = np.isfinite(t_seg) & np.isfinite(y_seg)
            t_seg = t_seg[mm]
            y_seg = _preprocess_prediction_for_correlation(method.name, y_seg[mm])
            if t_seg.size < 2:
                pred_concat[method.name].append(np.full(ref_trace.shape, np.nan, dtype=np.float64))
                continue
            pred_smoothed = smooth_prediction(y_seg, float(method.sampling_rate), sigma_ms=float(sigma_ms))
            pred_aligned = resample_prediction_to_reference(
                t_seg,
                pred_smoothed,
                ref_time,
                fs_est=float(method.sampling_rate),
            )
            pred_concat[method.name].append(np.asarray(pred_aligned, dtype=np.float64).ravel())

    if not ref_concat:
        return {m.name: float("nan") for m in methods}
    ref_all = np.concatenate(ref_concat)
    correlations: Dict[str, float] = {}
    for method in methods:
        preds = pred_concat.get(method.name, [])
        if not preds:
            correlations[method.name] = float("nan")
            continue
        pred_all = np.concatenate(preds)
        correlations[method.name] = _pearson(ref_all, pred_all)
    return correlations


def compute_trialwise_correlations_windowed(
    methods: Sequence[MethodResult],
    spike_times: np.ndarray,
    *,
    trial_windows: Sequence[Tuple[float, float]],
    reference_fs: float,
    sigma_ms: float = 50.0,
    binning: str = "linear",
) -> Dict[str, List[float]]:
    """
    Compute a correlation per method per trial window using start/end anchored reference grids.
    """
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    segments_by_method: Dict[str, List[slice]] = {}
    for method in methods:
        segments_by_method[method.name] = segment_indices(method.time_stamps, method.sampling_rate)

    out: Dict[str, List[float]] = {m.name: [] for m in methods}
    for idx, (start, end) in enumerate(trial_windows):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            for method in methods:
                out[method.name].append(float("nan"))
            continue
        ref_time, ref_trace = build_ground_truth_series(
            spike_times,
            float(start),
            float(end),
            reference_fs=float(reference_fs),
            sigma_ms=float(sigma_ms),
            binning=binning,
        )
        for method in methods:
            seg = _select_segment_for_window(
                segments_by_method.get(method.name, []),
                method.time_stamps,
                idx,
                float(start),
                float(end),
                n_windows=len(trial_windows),
            )
            if seg is None:
                out[method.name].append(float("nan"))
                continue
            t_seg = np.asarray(method.time_stamps, dtype=np.float64).ravel()[seg]
            y_seg = np.asarray(method.spike_prob, dtype=np.float64).ravel()[seg]
            mm = np.isfinite(t_seg) & np.isfinite(y_seg)
            t_seg = t_seg[mm]
            y_seg = _preprocess_prediction_for_correlation(method.name, y_seg[mm])
            if t_seg.size < 2:
                out[method.name].append(float("nan"))
                continue
            pred_smoothed = smooth_prediction(y_seg, float(method.sampling_rate), sigma_ms=float(sigma_ms))
            pred_aligned = resample_prediction_to_reference(
                t_seg,
                pred_smoothed,
                ref_time,
                fs_est=float(method.sampling_rate),
            )
            out[method.name].append(_pearson(ref_trace, pred_aligned))
    return out


def compute_trialwise_correlations(
    methods: Sequence[MethodResult],
    reference_time: np.ndarray,
    reference_trace: np.ndarray,
    *,
    trial_windows: Sequence[Tuple[float, float]],
    sigma_ms: float = 50.0,
) -> Dict[str, List[float]]:
    """
    Compute a correlation per method per trial window.

    This mirrors compute_correlations(), but returns a list of correlations (one per window)
    for each method. The prediction smoothing and resampling-to-reference are computed once
    per method and reused across windows.
    """
    reference_time = np.asarray(reference_time, dtype=float).ravel()
    reference_trace = np.asarray(reference_trace, dtype=float).ravel()
    if reference_time.shape != reference_trace.shape:
        raise ValueError(
            f"reference_time shape {reference_time.shape} does not match reference_trace {reference_trace.shape}"
        )

    window_masks: List[np.ndarray] = []
    for start, end in trial_windows:
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            window_masks.append(np.zeros(reference_time.shape, dtype=bool))
            continue
        window_masks.append((reference_time >= start) & (reference_time <= end))

    correlations: Dict[str, List[float]] = {}
    for method in methods:
        pred_smoothed = smooth_prediction(method.spike_prob, method.sampling_rate, sigma_ms=sigma_ms)
        aligned = resample_prediction_to_reference(
            method.time_stamps,
            pred_smoothed,
            reference_time,
            fs_est=method.sampling_rate,
        )
        valid_base = np.isfinite(aligned) & np.isfinite(reference_trace)
        values: List[float] = []
        for mask in window_masks:
            valid = valid_base & mask
            if valid.sum() < 2:
                values.append(float("nan"))
                continue
            x = reference_trace[valid]
            y = aligned[valid]
            x = x - np.mean(x)
            y = y - np.mean(y)
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            values.append(float(np.nan if denom == 0 else np.dot(x, y) / denom))
        correlations[method.name] = values
    return correlations
