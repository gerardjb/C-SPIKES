#!/usr/bin/env python3
"""
Utility script for comparing multiple spike inference methods on a single dataset.

The workflow mirrors the existing quick inference helpers in the project root:
  • load the first excitatory recording from the Janelia 8f dataset
  • optionally downsample traces to 30 Hz using the model_eval.boxcar_smoothing helpers
  • run PGAS, ENS2, and CASCADE inference backends
  • restrict PGAS to curated analysis windows defined in results/excitatory_time_stamp_edges.npy
  • smooth both ground-truth spikes and predicted spike probabilities with 50 ms kernels
  • compute Pearson correlations between the aligned, smoothed signals over the PGAS support only

The script is meant to be a starting point for broader sweeps across smoothing widths
and inference methods—configuration hooks are provided so future extensions can iterate
over alternative settings without rewriting the core orchestration.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import json
import hashlib
import datetime
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import scipy.io as sio

from c_spikes.utils import load_Janelia_data, unroll_mean_pgas_traj
from c_spikes.model_eval.boxcar_smoothing import downsample_from_dff
from c_spikes.model_eval.model_eval import smooth_prediction, smooth_spike_train


@dataclass
class TrialSeries:
    """Container for a single trial/segment of fluorescence data."""

    times: np.ndarray
    values: np.ndarray

    def current_fs(self) -> float:
        diffs = np.diff(self.times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            raise ValueError("Cannot estimate sampling rate for empty or degenerate trial.")
        return float(1.0 / np.median(diffs))


@dataclass
class MethodResult:
    """Standardised result bundle returned by each inference backend."""

    name: str
    time_stamps: np.ndarray
    spike_prob: np.ndarray
    sampling_rate: float
    metadata: Dict[str, object] = field(default_factory=dict)
    reconstruction: Optional[np.ndarray] = None
    discrete_spikes: Optional[np.ndarray] = None


CACHE_ROOT = Path("results") / "inference_cache"

PGAS_RESAMPLE_FS = 120.0
CASCADE_RESAMPLE_FS = 30.0
PGAS_MAX_SPIKES_PER_BIN = 1  # actual per-bin limit
PGAS_MAX_SPIKES_PARAM = PGAS_MAX_SPIKES_PER_BIN + 1  # analyzer expects +1 slack
PGAS_REFINE_COUNT_THRESHOLD = 0.8  # trigger refinement when MAP count nears limit
PGAS_REFINE_MIN_DURATION = 0.15  # seconds; ignore very short excursions
PGAS_REFINE_MAX_WINDOWS = 6
PGAS_REFINEMENT_MULTIPLIER = 2.0  # halves the bin width when refining
PGAS_REFINE_PADDING = 0.05  # seconds of context on each side of a window


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
    encoded = json.dumps(config_ser, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], config_ser


def hash_series(times: np.ndarray, values: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(times, dtype=np.float64).tobytes())
    h.update(np.asarray(values, dtype=np.float32).tobytes())
    return h.hexdigest()


def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr).astype(np.float32).tobytes()).hexdigest()


def get_cache_paths(method: str, dataset_tag: str, config_hash: str, ensure_dir: bool = False) -> Tuple[Path, Path]:
    cache_dir = CACHE_ROOT / method / dataset_tag
    if ensure_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{config_hash}.mat", cache_dir / f"{config_hash}.json"


def build_constants_cache_path(base_constants: Path, tokens: Sequence[str]) -> Path:
    cache_dir = CACHE_ROOT / "pgas_constants"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(tokens)
    return cache_dir / f"{base_constants.stem}_{suffix}{base_constants.suffix}"


def prepare_constants_with_params(
    base_constants: Path,
    *,
    maxspikes: int,
    bm_sigma: Optional[float] = None,
    substeps_per_frame: Optional[int] = None,
    physics_frequency_hz: Optional[float] = None,
    min_substeps: Optional[int] = None,
    time_integrated_observations: Optional[bool] = None,
    c0_is_first_y: Optional[bool] = None,
) -> Path:
    base_constants = Path(base_constants)
    tokens = [f"ms{maxspikes}"]
    if bm_sigma is not None:
        tokens.append(f"bm{format_tag_token(f'{bm_sigma:.4g}')}")
    if substeps_per_frame is not None:
        tokens.append(f"spf{substeps_per_frame}")
    elif physics_frequency_hz is not None:
        tokens.append(f"pfreq{format_tag_token(f'{physics_frequency_hz:g}')}")
    if min_substeps is not None:
        tokens.append(f"mins{min_substeps}")
    if time_integrated_observations is not None:
        tokens.append("int" if time_integrated_observations else "inst")
    if c0_is_first_y:
        tokens.append("c0y")
    target_path = build_constants_cache_path(base_constants, tokens)
    if target_path.exists():
        return target_path
    with base_constants.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("MCMC", {})["maxspikes"] = int(maxspikes)
    if bm_sigma is not None:
        data.setdefault("BM", {})["bm_sigma"] = float(bm_sigma)
    if c0_is_first_y is not None:
        data.setdefault("MCMC", {})["c0_is_first_y"] = bool(c0_is_first_y)
    if any(
        val is not None
        for val in (substeps_per_frame, physics_frequency_hz, min_substeps, time_integrated_observations)
    ):
        sub_cfg = data.setdefault("substepping", {})
        if substeps_per_frame is not None:
            sub_cfg["substeps_per_frame"] = int(substeps_per_frame)
        if physics_frequency_hz is not None:
            sub_cfg["physics_frequency_hz"] = float(physics_frequency_hz)
        if min_substeps is not None:
            sub_cfg["min_substeps"] = int(min_substeps)
        if time_integrated_observations is not None:
            sub_cfg["time_integrated_observations"] = bool(time_integrated_observations)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return target_path


def maxspikes_for_rate(target_fs: Optional[float], native_fs: float) -> int:
    """
    Choose a PGAS maxspikes setting that grants extra headroom when traces are heavily
    downsampled (e.g., 10 Hz). Allowing additional spikes per bin prevents the sampler
    from flattening bursty firing rates, which in turn improves downstream correlations.
    """
    if target_fs is None or np.isclose(target_fs, native_fs):
        return PGAS_MAX_SPIKES_PARAM
    if target_fs <= 0:
        raise ValueError("target_fs must be positive when provided.")

    ratio = max(native_fs / target_fs, 1.0)
    dynamic_limit = max(PGAS_MAX_SPIKES_PARAM + 1, int(math.ceil(ratio)) + 1)

    if np.isclose(target_fs, 30.0, atol=1e-1):
        # 30 Hz bins rarely require very high spike counts but still benefit from a
        # little slack beyond the single-spike-per-bin default.
        return max(4, int(math.ceil(ratio * 0.5)) + 1)
    if np.isclose(target_fs, 10.0, atol=1e-1):
        # 10 Hz smoothing aggregates 100 ms of activity; permit substantially more spikes.
        return max(9, dynamic_limit)
    return dynamic_limit


def compute_robust_diff_std(
    times: np.ndarray,
    values: np.ndarray,
    clip_percentiles: Tuple[float, float] = (5.0, 95.0),
) -> float:
    """
    Estimate the standard deviation of first differences using a robust MAD-based scale.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size < 2 or values.size < 2:
        return 0.0
    order = np.argsort(times)
    diffs = np.diff(values[order])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    if clip_percentiles is not None:
        lo, hi = np.percentile(diffs, clip_percentiles)
        mask = (diffs >= lo) & (diffs <= hi)
        if mask.any():
            diffs = diffs[mask]
    median = np.median(diffs)
    mad = np.median(np.abs(diffs - median))
    if mad <= 0:
        return float(np.std(diffs)) if diffs.size > 0 else 0.0
    return float(1.4826 * mad)


def build_low_activity_mask(
    sample_times: np.ndarray,
    spike_times: np.ndarray,
    exclusion: float,
) -> np.ndarray:
    """Mark samples that lie farther than `exclusion` seconds from any spike."""
    times = np.asarray(sample_times, dtype=np.float64)
    if times.size == 0 or exclusion <= 0:
        return np.ones_like(times, dtype=bool)
    spikes = np.asarray(spike_times, dtype=np.float64).ravel()
    if spikes.size == 0:
        return np.ones_like(times, dtype=bool)
    spikes = np.sort(spikes)
    idx = np.searchsorted(spikes, times)
    prev = np.full(times.shape, -np.inf)
    next_ = np.full(times.shape, np.inf)
    mask_prev = idx > 0
    prev[mask_prev] = spikes[idx[mask_prev] - 1]
    mask_next = idx < spikes.size
    next_[mask_next] = spikes[idx[mask_next]]
    dist_prev = np.abs(times - prev)
    dist_next = np.abs(next_ - times)
    min_dist = np.minimum(dist_prev, dist_next)
    return min_dist >= exclusion


def derive_bm_sigma(
    times: np.ndarray,
    values: np.ndarray,
    target_fs: float,
    scale_factor: float = 0.25,
    min_sigma: float = 5e-4,
    max_sigma: float = 5e-2,
) -> float:
    """
    Map the observed per-step diff scale to the Brownian motion sigma (per sqrt-second).
    """
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")
    diff_std = compute_robust_diff_std(times, values)
    if diff_std <= 0:
        return float(min_sigma)
    dt = 1.0 / target_fs
    bm_sigma = scale_factor * diff_std / math.sqrt(dt)
    return float(np.clip(bm_sigma, min_sigma, max_sigma))


def save_method_cache(
    method: str,
    dataset_tag: str,
    result: MethodResult,
    config: Dict[str, Any],
    trace_hash: str,
) -> None:
    config_hash, config_ser = compute_config_signature(config)
    mat_path, meta_path = get_cache_paths(method, dataset_tag, config_hash, ensure_dir=True)
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
    config: Dict[str, Any],
    trace_hash: str,
    *,
    allow_mismatched_trace: bool = False,
) -> Optional[MethodResult]:
    config_hash, config_ser = compute_config_signature(config)
    mat_path, meta_path = get_cache_paths(method, dataset_tag, config_hash, ensure_dir=False)
    candidates: List[Tuple[Path, Path]] = []
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
        result = MethodResult(
            name=method,
            time_stamps=time_stamps,
            spike_prob=spike_prob,
            sampling_rate=float(meta.get("sampling_rate", 0.0)),
            metadata=dict(meta.get("metadata", {})),
            reconstruction=reconstruction,
            discrete_spikes=discrete,
        )
        result.metadata.setdefault("cache_key", meta.get("cache_key"))
        result.metadata.setdefault("cached", True)
        result.metadata.setdefault("cache_tag", meta.get("metadata", {}).get("cache_tag", dataset_tag))
        return result
    return None


def get_pgas_time_grid_path(output_root: Path, dataset_tag: str) -> Path:
    """Resolve the path used to persist the per-trial time grid for PGAS runs."""
    return output_root / f"time_grid_{dataset_tag}.npz"


def persist_pgas_time_grid(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
) -> Path:
    """
    Persist the exact timestamp grid fed into PGAS so cached trajectories can be
    reconstructed later even if preprocessing changes.
    """
    grid_path = get_pgas_time_grid_path(output_root, dataset_tag)
    payload: Dict[str, np.ndarray] = {
        "trial_count": np.asarray([len(trials)], dtype=np.int64),
        "sample_counts": np.asarray([len(trial.times) for trial in trials], dtype=np.int64),
        "created_utc": np.asarray(datetime.datetime.utcnow().isoformat() + "Z"),
    }
    for idx, trial in enumerate(trials):
        key = f"trial_{idx:04d}"
        payload[key] = np.asarray(trial.times, dtype=np.float64)
    np.savez_compressed(grid_path, **payload)
    return grid_path


def load_pgas_time_grid(dataset_tag: str, output_root: Path) -> Optional[List[np.ndarray]]:
    """
    Load the persisted PGAS time grid if present.

    Returns:
        List of per-trial timestamp arrays in acquisition order, or None if unavailable.
    """
    grid_path = get_pgas_time_grid_path(output_root, dataset_tag)
    if not grid_path.exists():
        return None
    try:
        with np.load(grid_path, allow_pickle=False) as data:
            if "trial_count" not in data:
                raise ValueError(f"Malformed PGAS grid file: {grid_path}")
            trial_count = int(np.asarray(data["trial_count"]).ravel()[0])
            times: List[np.ndarray] = []
            for idx in range(trial_count):
                key = f"trial_{idx:04d}"
                if key not in data:
                    raise ValueError(
                        f"PGAS grid file {grid_path} missing entry '{key}'. "
                        "Re-run PGAS to refresh cached outputs."
                    )
                arr = np.asarray(data[key], dtype=np.float64)
                times.append(arr.copy())
            return times
    except Exception as exc:
        raise RuntimeError(f"Failed to load PGAS time grid from {grid_path}") from exc


def build_trials_from_time_grid(time_grid: Sequence[np.ndarray]) -> List[TrialSeries]:
    """
    Create placeholder TrialSeries objects from a stored time grid so downstream
    loaders can rebuild PGAS trajectories without needing the original traces.
    """
    return [
        TrialSeries(times=times.copy(), values=np.zeros_like(times, dtype=np.float64))
        for times in time_grid
    ]


def extract_trials(time_stamps: np.ndarray, traces: np.ndarray) -> List[TrialSeries]:
    """Split raw matrices into per-trial series while dropping NaNs."""
    trials: List[TrialSeries] = []
    for idx in range(time_stamps.shape[0]):
        t = np.asarray(time_stamps[idx], dtype=np.float64)
        y = np.asarray(traces[idx], dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        if t.size == 0:
            continue
        trials.append(TrialSeries(times=t, values=y))
    if not trials:
        raise RuntimeError("No valid trials found in the provided dataset.")
    return trials


def downsample_trials(trials: Sequence[TrialSeries], target_fs: float) -> List[TrialSeries]:
    """Create 30 Hz versions of each trial using the provided boxcar smoothing helper."""
    ds_trials: List[TrialSeries] = []
    for trial in trials:
        fs_current = trial.current_fs()
        _, dff_ds, t_ds = downsample_from_dff(
            trial.values,
            fs=fs_current,
            fs_target=target_fs,
            F0=1.0,
            recompute_dff=True,
            baseline_win_s=60,
            baseline_perc=8,
            t0=float(trial.times[0]),
        )
        ds_trials.append(
            TrialSeries(
                times=np.asarray(t_ds, dtype=np.float64),
                values=np.asarray(dff_ds, dtype=np.float64),
            )
        )
    return ds_trials


def trim_trials_by_edges(
    trials: Sequence[TrialSeries],
    edges: np.ndarray,
    tolerance: float = 1e-6,
) -> List[TrialSeries]:
    """
    Restrict each trial to the time window specified by edges[idx] = [start, end].

    Args:
        trials: Sequence of TrialSeries objects (ordered by acquisition index).
        edges: Array of shape (len(trials), 2) containing start/end timestamps in seconds.
        tolerance: Small slack (seconds) to include samples bordering the window.

    Returns:
        List[TrialSeries]: Trimmed trials aligned with the provided edges.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.shape[0] != len(trials) or edges.shape[1] != 2:
        raise ValueError(
            f"Expected edges with shape (n_trials, 2); got {edges.shape} for {len(trials)} trials."
        )

    trimmed: List[TrialSeries] = []
    for idx, (trial, (start, end)) in enumerate(zip(trials, edges)):
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError(f"Non-finite window bounds ({start}, {end}) for trial {idx}.")
        if end <= start:
            raise ValueError(f"Window end must exceed start for trial {idx}: ({start}, {end}).")

        mask = (trial.times >= start - tolerance) & (trial.times <= end + tolerance)
        if not mask.any():
            raise ValueError(
                f"No samples within window ({start}, {end}) for trial {idx}; check edges resolution."
            )
        trimmed.append(
            TrialSeries(times=trial.times[mask].copy(), values=trial.values[mask].copy())
        )

    return trimmed


def mean_downsample_trace(times: np.ndarray, values: np.ndarray, target_fs: float) -> TrialSeries:
    """Simple bin-averaging downsample without re-baselining."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size == 0:
        return TrialSeries(times=times, values=values)
    dt = np.median(np.diff(times))
    fs = 1.0 / dt
    if fs <= target_fs:
        return TrialSeries(times=times.copy(), values=values.copy())
    ratio = fs / target_fs
    # Treat near-integer ratios as integer for binning
    B = int(round(ratio))
    if not np.isclose(ratio, B, atol=1e-6):
        # fall back to linear interpolation
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
    """Resample a single trial to the requested sampling rate via binning or interpolation."""
    if target_fs <= 0:
        raise ValueError("target_fs must be positive")
    times = np.asarray(trial.times, dtype=np.float64)
    values = np.asarray(trial.values, dtype=np.float64)
    if times.size <= 1:
        return TrialSeries(times=times.copy(), values=values.copy())

    current_fs = compute_sampling_rate(times)
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
    """Apply resample_trial_to_fs to a collection of trials."""
    return [resample_trial_to_fs(trial, target_fs) for trial in trials]


def find_refinement_windows(
    result: MethodResult,
    threshold: float = PGAS_REFINE_COUNT_THRESHOLD,
    min_duration: float = PGAS_REFINE_MIN_DURATION,
    max_windows: int = PGAS_REFINE_MAX_WINDOWS,
) -> List[Dict[str, float]]:
    """Locate contiguous intervals where the MAP spikes exceed the allowed per-bin limit."""
    if result.discrete_spikes is not None:
        values = np.asarray(result.discrete_spikes, dtype=np.float64)
    else:
        values = np.asarray(result.spike_prob, dtype=np.float64)
    times = np.asarray(result.time_stamps, dtype=np.float64)
    if times.size == 0 or values.size != times.size:
        return []
    mask = values >= threshold
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    segments: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for current in idx[1:]:
        if current == prev + 1:
            prev = current
            continue
        segments.append((start, prev))
        start = current
        prev = current
    segments.append((start, prev))

    windows: List[Dict[str, float]] = []
    for seg_start, seg_end in segments:
        t_start = float(times[seg_start])
        t_end = float(times[seg_end])
        if t_end - t_start < min_duration:
            continue
        peak = float(values[seg_start : seg_end + 1].max())
        windows.append({"start": t_start, "end": t_end, "peak": peak})
        if len(windows) >= max_windows:
            break
    return windows


def extract_window_series(
    trials: Sequence[TrialSeries], start: float, end: float
) -> TrialSeries:
    """Collect samples within [start, end] from a list of trials."""
    if start >= end:
        raise ValueError("window start must precede end")
    segments: List[TrialSeries] = []
    for trial in trials:
        mask = (trial.times >= start) & (trial.times <= end)
        if mask.any():
            segments.append(
                TrialSeries(times=trial.times[mask].copy(), values=trial.values[mask].copy())
            )
    if not segments:
        raise ValueError("No samples found within refinement window.")
    times, values = flatten_trials(segments)
    if times.size < 2:
        raise ValueError("Insufficient samples to refine window.")
    return TrialSeries(times=times, values=values)


def merge_pgas_refined_segments(
    base_result: MethodResult,
    segments: List[Dict[str, object]],
) -> MethodResult:
    """Replace coarse PGAS outputs within each refined window using higher-rate results."""
    if not segments:
        return base_result
    base_times = np.asarray(base_result.time_stamps, dtype=np.float64)
    keep_mask = np.ones(base_times.shape, dtype=bool)
    for segment in segments:
        window = segment["window"]  # type: ignore[index]
        start = float(window["start"])  # type: ignore[index]
        end = float(window["end"])  # type: ignore[index]
        keep_mask &= ~((base_times >= start) & (base_times <= end))

    ordered_segments = sorted(segments, key=lambda seg: float(seg["window"]["start"]))  # type: ignore[index]
    time_blocks = [base_times[keep_mask]] + [np.asarray(seg["result"].time_stamps) for seg in ordered_segments]  # type: ignore[index]
    combined_times = np.concatenate(time_blocks)
    order = np.argsort(combined_times)

    def combine_attr(attr: str) -> Optional[np.ndarray]:
        parts: List[np.ndarray] = []
        base_attr = getattr(base_result, attr)
        if base_attr is not None:
            parts.append(np.asarray(base_attr)[keep_mask])
        for seg in ordered_segments:
            seg_attr = getattr(seg["result"], attr)  # type: ignore[index]
            if seg_attr is not None:
                parts.append(np.asarray(seg_attr))
        if not parts:
            return None
        combined = np.concatenate(parts)
        return combined[order]

    combined_time_axis = combined_times[order]
    spike_prob = combine_attr("spike_prob")
    if spike_prob is None:
        raise ValueError("PGAS refinement merge produced empty spike_prob array.")

    merged_result = MethodResult(
        name=base_result.name,
        time_stamps=combined_time_axis,
        spike_prob=spike_prob,
        sampling_rate=compute_sampling_rate(combined_time_axis),
        metadata=dict(base_result.metadata),
        reconstruction=combine_attr("reconstruction"),
        discrete_spikes=combine_attr("discrete_spikes"),
    )
    return merged_result


def resample_method_to_grid(method: MethodResult, target_time: np.ndarray) -> MethodResult:
    """
    Resample a MethodResult onto a specified time grid while preserving metadata.

    This is useful when adaptive refinement has produced a denser time axis but
    evaluation (e.g. correlations, summaries) should be performed on a common
    base PGAS grid.
    """
    target_time = np.asarray(target_time, dtype=np.float64)

    def _resample(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        return resample_prediction_to_reference(
            np.asarray(method.time_stamps, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            target_time,
            fs_est=method.sampling_rate,
        )

    spike_prob_rs = _resample(method.spike_prob)
    if spike_prob_rs is None:
        raise ValueError("Cannot resample MethodResult with empty spike_prob.")

    reconstruction_rs = _resample(method.reconstruction)
    discrete_rs = _resample(method.discrete_spikes)

    meta = dict(method.metadata or {})
    sampling_rate = compute_sampling_rate(target_time) if target_time.size > 1 else method.sampling_rate

    return MethodResult(
        name=method.name,
        time_stamps=target_time,
        spike_prob=spike_prob_rs,
        sampling_rate=sampling_rate,
        metadata=meta,
        reconstruction=reconstruction_rs,
        discrete_spikes=discrete_rs,
    )

def format_tag_token(value: str) -> str:
    """Collapse a descriptive token into a filesystem-friendly identifier."""
    return value.replace(" ", "_").replace(".", "p")


def load_pgas_method_result(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    burnin: int,
    metadata: Optional[Dict[str, object]] = None,
) -> MethodResult:
    """Reconstruct a MethodResult from PGAS trajectory files saved on disk."""
    traces = load_pgas_component_series(trials, dataset_tag, output_root, burnin)
    fs_est = compute_sampling_rate(traces["time_stamps"])
    meta = dict(metadata) if metadata else {}
    return MethodResult(
        name="pgas",
        time_stamps=traces["time_stamps"],
        spike_prob=traces["spikes_mean"],
        sampling_rate=fs_est,
        metadata=meta,
        reconstruction=traces["calcium_mean"],
        discrete_spikes=traces["spikes_map"],
    )


def load_pgas_component_series(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    burnin: int,
) -> Dict[str, np.ndarray]:
    """Load PGAS trajectory components (spikes, baseline, calcium, MAP) for QC plots."""
    spike_segments: List[np.ndarray] = []
    time_segments: List[np.ndarray] = []
    baseline_segments: List[np.ndarray] = []
    calcium_segments: List[np.ndarray] = []
    burst_segments: List[np.ndarray] = []
    map_segments: List[np.ndarray] = []
    for trial_idx, trial in enumerate(trials):
        tag = f"{dataset_tag}_trial{trial_idx}"
        dat_file = output_root / f"traj_samples_{tag}.dat"
        log_file = output_root / f"logp_{tag}.dat"
        if not dat_file.exists() or not log_file.exists():
            raise FileNotFoundError(
                f"Missing PGAS output files for tag '{tag}'. Expected {dat_file} and {log_file}."
            )
        (
            burst_mean,
            baseline_mean,
            spikes_mean,
            C_mean,
            spikes_map,
        ) = unroll_mean_pgas_traj(str(dat_file), str(log_file), burnin=burnin)
        time_segments.append(trial.times.copy())
        spike_segments.append(np.asarray(spikes_mean, dtype=np.float64))
        baseline_segments.append(np.asarray(baseline_mean, dtype=np.float64))
        calcium_segments.append(np.asarray(C_mean + baseline_mean, dtype=np.float64))
        burst_segments.append(np.asarray(burst_mean, dtype=np.float64))
        map_segments.append(np.asarray(spikes_map, dtype=np.float64))

    def align_and_concat(values: Sequence[np.ndarray], label: str) -> Tuple[np.ndarray, np.ndarray]:
        aligned_times: List[np.ndarray] = []
        aligned_vals: List[np.ndarray] = []
        for idx, (times_arr, vals_arr) in enumerate(zip(time_segments, values)):
            n = min(times_arr.size, vals_arr.size)
            if n == 0:
                continue
            if times_arr.size != vals_arr.size:
                print(
                    f"[PGAS QC] Warning: truncating {label} segment for trial {idx} "
                    f"(time={times_arr.size}, values={vals_arr.size})."
                )
            aligned_times.append(times_arr[:n])
            aligned_vals.append(vals_arr[:n])
        if not aligned_times:
            raise ValueError(f"No samples available for PGAS {label} traces.")
        return np.concatenate(aligned_times), np.concatenate(aligned_vals)

    times, spikes = align_and_concat(spike_segments, "spike")
    baseline_times, baseline = align_and_concat(baseline_segments, "baseline")
    calcium_times, calcium = align_and_concat(calcium_segments, "calcium")
    map_times, map_values = align_and_concat(map_segments, "map")
    burst_times, burst = align_and_concat(burst_segments, "burst")

    for label, arr_times in {
        "baseline": baseline_times,
        "calcium": calcium_times,
        "map": map_times,
        "burst": burst_times,
    }.items():
        if arr_times.shape != times.shape or not np.allclose(arr_times, times):
            raise ValueError(f"PGAS {label} timestamps do not align with spike traces.")

    return {
        "time_stamps": times,
        "spikes_mean": spikes,
        "baseline_mean": baseline,
        "burst_mean": burst,
        "calcium_mean": calcium,
        "spikes_map": map_values,
    }


def flatten_trials(trials: Sequence[TrialSeries]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate trials into a single time/value series sorted by acquisition time."""
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


def run_pgas_inference(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    constants_file: Path,
    gparam_file: Path,
    niter: int = 200,
    burnin: int = 100,
    recompute: bool = True,
    verbose: bool = False,
) -> MethodResult:
    """Execute the PGAS bound analyzer on each trial and collect the mean spike trajectories."""
    try:
        pgas = importlib.import_module("c_spikes.pgas.pgas_bound")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PGAS module not found. Build the c_spikes.pgas extension before running this script."
        ) from exc

    output_root.mkdir(parents=True, exist_ok=True)
    trial_list = list(trials)

    if recompute:
        for trial_idx, trial in enumerate(trial_list):
            tag = f"{dataset_tag}_trial{trial_idx}"
            analyzer = pgas.Analyzer(
                time=np.ascontiguousarray(trial.times, dtype=np.float64),
                data=np.ascontiguousarray(trial.values, dtype=np.float64),
                constants_file=str(constants_file),
                output_folder=str(output_root),
                column=1,
                tag=tag,
                niter=niter,
                trainedPriorFile="",
                append=False,
                trim=1,
                verbose=verbose,
                gtSpikes=np.zeros(0, dtype=np.float64),
                has_trained_priors=False,
                has_gtspikes=False,
                maxlen=int(trial.values.size),
                Gparam_file=str(gparam_file),
                seed=2 + trial_idx,
            )
            analyzer.run()
        persist_pgas_time_grid(trial_list, dataset_tag, output_root)
        trials_for_reconstruction: Sequence[TrialSeries] = trial_list
    else:
        persisted_grid = load_pgas_time_grid(dataset_tag, output_root)
        if persisted_grid:
            trials_for_reconstruction = build_trials_from_time_grid(persisted_grid)
        else:
            grid_path = get_pgas_time_grid_path(output_root, dataset_tag)
            print(
                f"[PGAS] Warning: No persisted time grid at {grid_path}; "
                "falling back to provided trials."
            )
            trials_for_reconstruction = trial_list

    return load_pgas_method_result(
        trials=trials_for_reconstruction,
        dataset_tag=dataset_tag,
        output_root=output_root,
        burnin=burnin,
        metadata={"burnin": burnin, "niter": niter, "output_root": str(output_root)},
    )


def refine_pgas_result(
    base_result: MethodResult,
    trials_resampled: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    constants_file: Path,
    gparam_file: Path,
    niter: int,
    burnin: int,
    base_fs: float,
    threshold: float = PGAS_REFINE_COUNT_THRESHOLD,
    min_duration: float = PGAS_REFINE_MIN_DURATION,
    max_windows: int = PGAS_REFINE_MAX_WINDOWS,
) -> MethodResult:
    """Run PGAS on high-priority windows using a finer grid and merge the results."""
    windows = find_refinement_windows(
        base_result, threshold=threshold, min_duration=min_duration, max_windows=max_windows
    )
    if not windows:
        return base_result

    min_time = min(trial.times[0] for trial in trials_resampled)
    max_time = max(trial.times[-1] for trial in trials_resampled)
    refined_fs = base_fs * PGAS_REFINEMENT_MULTIPLIER
    segments: List[Dict[str, object]] = []

    for idx, window in enumerate(windows):
        start = max(window["start"] - PGAS_REFINE_PADDING, min_time)
        end = min(window["end"] + PGAS_REFINE_PADDING, max_time)
        try:
            window_series = extract_window_series(trials_resampled, start, end)
        except ValueError:
            continue
        refined_trial = resample_trial_to_fs(window_series, refined_fs)
        refined_tag = f"{dataset_tag}_refined_{idx}"
        trace_hash = hash_series(refined_trial.times, refined_trial.values)
        refine_config = {
            "niter": niter,
            "burnin": burnin,
            "constants_file": str(constants_file),
            "gparam_file": str(gparam_file),
            "refined_window": [float(window["start"]), float(window["end"])],
            "input_resample_fs": refined_fs,
            "adaptive_refine": True,
        }
        cached = load_method_cache("pgas", refined_tag, refine_config, trace_hash)
        if cached:
            refined_result = cached
            refined_result.metadata.setdefault("refined_window", refine_config["refined_window"])
            refined_result.metadata.setdefault("input_resample_fs", refined_fs)
            refined_result.metadata.setdefault("cache_tag", refined_tag)
            from_cache = True
        else:
            refined_result = run_pgas_inference(
                trials=[refined_trial],
                dataset_tag=refined_tag,
                output_root=output_root,
                constants_file=constants_file,
                gparam_file=gparam_file,
                niter=niter,
                burnin=burnin,
                recompute=True,
            )
            refined_result.metadata.setdefault("niter", niter)
            refined_result.metadata.setdefault("burnin", burnin)
            refined_result.metadata.setdefault("config", ensure_serializable(refine_config))
            refined_result.metadata.setdefault("input_resample_fs", refined_fs)
            refined_result.metadata.setdefault("refined_window", refine_config["refined_window"])
            refined_result.metadata["cache_tag"] = refined_tag
            save_method_cache("pgas", refined_tag, refined_result, refine_config, trace_hash)
            from_cache = False
        # Store padded window bounds for downstream use/metadata
        window_padded = dict(window)
        window_padded["start"] = float(start)
        window_padded["end"] = float(end)
        segments.append(
            {"window": window_padded, "result": refined_result, "cache_tag": refined_tag, "from_cache": from_cache}
        )

    if not segments:
        return base_result

    base_times = np.asarray(base_result.time_stamps, dtype=np.float64)
    spike_prob_base = np.asarray(base_result.spike_prob, dtype=np.float64)
    recon_base = (
        None if base_result.reconstruction is None else np.asarray(base_result.reconstruction, dtype=np.float64)
    )
    # Keep discrete_spikes on the base grid unchanged; refinement is treated as
    # a spike-probability / reconstruction improvement only.

    # Project each refined window back onto the base PGAS grid.
    for seg in segments:
        window = seg["window"]  # type: ignore[index]
        refined = seg["result"]  # type: ignore[index]
        start = float(window["start"])
        end = float(window["end"])
        mask = (base_times >= start) & (base_times <= end)
        if not mask.any():
            continue
        # Resample refined spike_prob onto the base grid, then overwrite within the window.
        refined_prob_on_base = resample_prediction_to_reference(
            np.asarray(refined.time_stamps, dtype=np.float64),
            np.asarray(refined.spike_prob, dtype=np.float64),
            base_times,
            fs_est=refined.sampling_rate,
        )
        spike_prob_base[mask] = refined_prob_on_base[mask]

        if recon_base is not None and refined.reconstruction is not None:
            refined_recon_on_base = resample_prediction_to_reference(
                np.asarray(refined.time_stamps, dtype=np.float64),
                np.asarray(refined.reconstruction, dtype=np.float64),
                base_times,
                fs_est=refined.sampling_rate,
            )
            recon_base[mask] = refined_recon_on_base[mask]

    merged = MethodResult(
        name=base_result.name,
        time_stamps=base_times,
        spike_prob=spike_prob_base,
        sampling_rate=compute_sampling_rate(base_times) if base_times.size > 1 else base_result.sampling_rate,
        metadata=dict(base_result.metadata),
        reconstruction=recon_base,
        discrete_spikes=base_result.discrete_spikes,
    )
    refined_info: List[Dict[str, object]] = []
    for seg in segments:
        window = dict(seg["window"])  # type: ignore[arg-type]
        window.update(
            {
                "cache_tag": seg["cache_tag"],
                "sampling_rate": getattr(seg["result"], "sampling_rate", refined_fs),
                "from_cache": seg["from_cache"],
            }
        )
        refined_info.append(window)
    merged.metadata["pgas_refined_windows"] = refined_info
    merged.metadata["adaptive_refine"] = True
    merged.metadata["refine_threshold"] = threshold
    merged.metadata["refine_multiplier"] = PGAS_REFINEMENT_MULTIPLIER
    merged.metadata.setdefault("input_resample_fs", base_fs)
    return merged


def run_ens2_inference(
    raw_time_stamps: np.ndarray,
    raw_traces: np.ndarray,
    dataset_tag: str,
    pretrained_dir: Path,
    neuron_type: str = "Exc",
) -> MethodResult:
    """Reproduce the ENS2 quick inference workflow and return spike probabilities."""
    try:
        ens2_module = importlib.import_module("c_spikes.ens2.ENS2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ENS2 module not found. Ensure the upstream ENS2 package is available on PYTHONPATH."
        ) from exc

    import torch  # Imported lazily to keep CUDA/Torch requirements optional
    try:  # torch>=2.6 exposes serialization helpers; older versions simply skip this block
        from torch.serialization import add_safe_globals  # type: ignore
    except ImportError:  # pragma: no cover
        add_safe_globals = None  # type: ignore
    try:
        from tqdm.auto import trange
    except ImportError:  # pragma: no cover - tqdm is optional here
        def trange(n, **kwargs):
            return range(n)

    # Mirror the quick script's alias so state_dict loading succeeds
    sys.modules.setdefault("ENS2", ens2_module)
    ens2 = ens2_module.ENS2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ens2.DEVICE = device

    pretrained_dir = pretrained_dir.expanduser()
    state_dict_path: Path
    if neuron_type.lower().startswith("exc"):
        state_dict_path = pretrained_dir / "exc_ens2_pub.pt"
    else:
        state_dict_path = pretrained_dir / "inh_ens2_pub.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"ENS2 checkpoint not found: {state_dict_path}")

    if add_safe_globals and hasattr(ens2_module, "UNet"):
        try:
            add_safe_globals([ens2_module.UNet])  # allow the serialized class when weights_only=True
        except Exception:
            pass

    load_kwargs = {"map_location": torch.device(device)}
    try:
        checkpoint = torch.load(state_dict_path, weights_only=False, **load_kwargs)
    except TypeError:  # older torch that lacks weights_only keyword
        checkpoint = torch.load(state_dict_path, **load_kwargs)

    state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    # Determine per-trial start times so we can restore absolute timing later
    trial_starts = raw_time_stamps[:, 0]
    trial_duration = float(raw_time_stamps[0, -1] - raw_time_stamps[0, 0])
    test_data = ens2_module.compile_test_data(raw_traces, trial_duration)

    time_segments: List[np.ndarray] = []
    rate_segments: List[np.ndarray] = []
    discrete_segments: List[np.ndarray] = []
    event_counts: List[int] = []
    frame_rates: List[float] = []

    for trial_idx in trange(len(test_data), desc="ENS2 inference", leave=False):
        dff_segment = test_data[trial_idx]["dff_resampled_segment"]
        _, temp_pd_rate, temp_pd_spike, temp_pd_event = ens2.predict(dff_segment, state_dict=state_dict)

        fluo_times = test_data[trial_idx]["fluo_times_resampled"]
        abs_time = np.asarray(fluo_times + trial_starts[trial_idx], dtype=np.float64).ravel()
        rate_values = np.asarray(temp_pd_rate, dtype=np.float64).ravel()
        discrete_values = np.asarray(temp_pd_spike, dtype=np.float64).ravel()

        time_segments.append(abs_time)
        rate_segments.append(rate_values)
        discrete_segments.append(discrete_values)
        event_counts.append(len(temp_pd_event))
        frame_rates.append(float(test_data[trial_idx]["frame_rate_resampled"]))

    times, rates = flatten_trials(
        [TrialSeries(times=t, values=s) for t, s in zip(time_segments, rate_segments)]
    )
    _, discrete = flatten_trials(
        [TrialSeries(times=t, values=s) for t, s in zip(time_segments, discrete_segments)]
    )
    fs_est = compute_sampling_rate(times)

    return MethodResult(
        name="ens2",
        time_stamps=times,
        spike_prob=rates,
        sampling_rate=fs_est,
        metadata={
            "device": device,
            "checkpoint": str(state_dict_path),
            "neuron_type": neuron_type,
            "event_counts": event_counts,
            "frame_rates": frame_rates,
        },
        discrete_spikes=discrete,
    )


def run_cascade_inference(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    model_folder: Path,
    model_name: str = "Cascade_Universal_30Hz",
) -> MethodResult:
    """Execute the CASCADE predictor on the provided trials."""
    from c_spikes.cascade2p import cascade
    from c_spikes.cascade2p.utils_discrete_spikes import infer_discrete_spikes

    # Note*: this guards against the occasional trial where I guess the sampling rate was incorrectly set
    min_len = min(trial.values.size for trial in trials)
    if any(trial.values.size != min_len for trial in trials):
        aligned_trials = [
            TrialSeries(times=trial.times[:min_len], values=trial.values[:min_len])
            for trial in trials
        ]
    else:
        aligned_trials = list(trials)
    trace_matrix = np.vstack([trial.values for trial in aligned_trials])
    spike_prob = cascade.predict(
        model_name,
        trace_matrix,
        model_folder=str(model_folder),
        reuse_models=True,
    )
    time_matrix = np.vstack([trial.times for trial in aligned_trials])

    approximations, spike_lists = infer_discrete_spikes(
        spike_prob, model_name, model_folder=str(model_folder)
    )
    discrete_matrix = np.zeros_like(spike_prob, dtype=float)
    for neuron_idx, spike_list in enumerate(spike_lists):
        if spike_list is None:
            continue
        indices = np.asarray(spike_list, dtype=int)
        indices = indices[(indices >= 0) & (indices < discrete_matrix.shape[1])]
        discrete_matrix[neuron_idx, indices] += 1

    times, spikes = flatten_trials(
        [
            TrialSeries(times=time_matrix[idx], values=spike_prob[idx])
            for idx in range(trace_matrix.shape[0])
        ]
    )
    _, discrete = flatten_trials(
        [
            TrialSeries(times=time_matrix[idx], values=discrete_matrix[idx])
            for idx in range(trace_matrix.shape[0])
        ]
    )
    fs_est = compute_sampling_rate(times)

    return MethodResult(
        name="cascade",
        time_stamps=times,
        spike_prob=np.asarray(spikes, dtype=np.float64),
        sampling_rate=fs_est,
        metadata={"model_name": model_name, "model_folder": str(model_folder)},
        discrete_spikes=np.asarray(discrete, dtype=np.float64),
    )


def build_ground_truth_series(
    spike_times: np.ndarray,
    global_start: float,
    global_end: float,
    reference_fs: float,
    sigma_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a smoothed ground-truth spike-rate trace once for the entire recording."""
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    duration = max(0.0, global_end - global_start)
    within_bounds = spike_times[(spike_times >= global_start) & (spike_times <= global_end)]
    relative_spikes = within_bounds - global_start
    smoothed = smooth_spike_train(
        relative_spikes, sampling_rate=reference_fs, duration=duration, sigma_ms=sigma_ms
    )
    time_axis = (np.arange(smoothed.size, dtype=np.float64) + 0.5) / reference_fs + global_start
    return time_axis, smoothed


def segment_indices(times: np.ndarray, fs_est: float, gap_factor: float = 4.0) -> List[slice]:
    """Identify contiguous segments within a time series to avoid interpolating over gaps."""
    if times.size == 0:
        return []
    diffs = np.diff(times)
    gap_threshold = gap_factor / fs_est
    breakpoints = np.where(diffs > gap_threshold)[0]
    segments: List[slice] = []
    start = 0
    for bp in breakpoints:
        segments.append(slice(start, bp + 1))
        start = bp + 1
    segments.append(slice(start, times.size))
    return segments


def resample_prediction_to_reference(
    times: np.ndarray,
    values: np.ndarray,
    reference_time: np.ndarray,
    fs_est: float,
) -> np.ndarray:
    """Map a method's prediction onto the reference grid without bridging gaps."""
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
            # Single sample segment; assign to nearest reference index
            idx = np.argmin(np.abs(reference_time - seg_times[0]))
            result[idx] = seg_values[0]
            continue
        result[mask] = np.interp(reference_time[mask], seg_times, seg_values)
    return result


def compute_correlations(
    methods: Sequence[MethodResult],
    reference_time: np.ndarray,
    reference_trace: np.ndarray,
    sigma_ms: float = 50.0,
    windows: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Smooth predictions and compute Pearson correlations against the reference trace."""
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
            correlations[method.name] = np.nan
            continue
        x = reference_trace[valid]
        y = aligned[valid]
        x -= np.mean(x)
        y -= np.mean(y)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        correlations[method.name] = float(np.nan if denom == 0 else np.dot(x, y) / denom)
    return correlations


def plot_inference_overlay(
    raw_time: np.ndarray,
    raw_trace: np.ndarray,
    down_time: Optional[np.ndarray],
    down_trace: Optional[np.ndarray],
    methods: Sequence[MethodResult],
    spike_times: np.ndarray,
    windows: Optional[Sequence[Tuple[float, float]]] = None,
    pgas_time: Optional[np.ndarray] = None,
    pgas_reconstruction: Optional[np.ndarray] = None,
    reference_time: Optional[np.ndarray] = None,
    reference_trace: Optional[np.ndarray] = None,
    sigma_ms: float = 50.0,
) -> None:
    """Visualise raw fluorescence, inference traces (offset), and ground-truth spikes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(f"Matplotlib import failed ({exc}); skipping plot.")
        return

    if raw_time.size == 0:
        print("No raw data available for plotting.")
        return

    if windows:
        valid_mask = np.zeros(raw_time.shape, dtype=bool)
        for start, end in windows:
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            valid_mask |= (raw_time >= start) & (raw_time <= end)
        if not valid_mask.any():
            valid_mask = np.ones_like(raw_time, dtype=bool)
    else:
        valid_mask = np.ones_like(raw_time, dtype=bool)

    time_window = raw_time[valid_mask]
    trace_window = raw_trace[valid_mask]
    if time_window.size == 0:
        print("No samples fall within the selected window; skipping plot.")
        return

    window_start = time_window.min()
    window_end = time_window.max()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_window, trace_window, label="raw dF/F", color="black", linewidth=1.2)

    if down_time is not None and down_trace is not None and down_time.size > 0:
        down_mask = (down_time >= window_start) & (down_time <= window_end)
        down_time_window = down_time[down_mask]
        down_trace_window = down_trace[down_mask]
        if down_time_window.size > 0:
            ax.plot(
                down_time_window,
                down_trace_window,
                label="downsampled dF/F",
                color="tab:blue",
                linewidth=1.0,
                alpha=0.8,
            )

    if (
        pgas_time is not None
        and pgas_reconstruction is not None
        and pgas_time.size > 0
        and pgas_reconstruction.size > 0
    ):
        recon_mask = (pgas_time >= window_start) & (pgas_time <= window_end)
        recon_time_window = pgas_time[recon_mask]
        recon_values_window = pgas_reconstruction[recon_mask]
        if recon_time_window.size > 0:
            ax.plot(
                recon_time_window,
                recon_values_window,
                label="PGAS mean trace",
                color="tab:green",
                linewidth=1.0,
            )

    method_traces: List[Tuple[np.ndarray, np.ndarray, MethodResult]] = []
    for method in methods:
        mask = (method.time_stamps >= window_start) & (method.time_stamps <= window_end)
        if not mask.any():
            continue
        method_traces.append(
            (method.time_stamps[mask], method.spike_prob[mask], method)
        )

    if not method_traces:
        print("No inference traces overlap with the plotting window; skipping plot.")
        return

    raw_span = float(np.nanmax(trace_window) - np.nanmin(trace_window))
    offset_step = max(raw_span, 1.0)
    for _, values, _ in method_traces:
        if values.size == 0:
            continue
        offset_step = max(offset_step, float(np.nanmax(values) - np.nanmin(values) + 0.5))

    current_offset = offset_step
    for idx, (times, values, method) in enumerate(method_traces, start=1):
        ax.plot(
            times,
            values + current_offset,
            label=f"{method.name} (offset {idx})",
            linewidth=1.0,
        )
        if method.name.lower() == "pgas":
            if reference_time is not None and reference_trace is not None:
                ref_mask = (reference_time >= window_start) & (reference_time <= window_end)
                ref_mask &= np.isfinite(reference_trace)
                if ref_mask.any():
                    ax.plot(
                        reference_time[ref_mask],
                        reference_trace[ref_mask] + current_offset,
                        label="GT smoothed (50 ms)",
                        color="tab:red",
                        linewidth=1.0,
                        alpha=0.9,
                    )
                smoothed_full = smooth_prediction(method.spike_prob, method.sampling_rate, sigma_ms=sigma_ms)
                aligned_smoothed = resample_prediction_to_reference(
                    method.time_stamps,
                    smoothed_full,
                    reference_time,
                    method.sampling_rate,
                )
                smooth_mask = (reference_time >= window_start) & (reference_time <= window_end)
                smooth_mask &= np.isfinite(aligned_smoothed)
                if smooth_mask.any():
                    ax.plot(
                        reference_time[smooth_mask],
                        aligned_smoothed[smooth_mask] + current_offset,
                        label="PGAS smoothed (50 ms)",
                        color="tab:orange",
                        linewidth=1.0,
                        linestyle="--",
                    )
        current_offset += offset_step

    spikes = np.asarray(spike_times, dtype=float).ravel()
    if spikes.size > 0:
        spike_mask = (spikes >= window_start) & (spikes <= window_end)
        for spk_time in spikes[spike_mask]:
            ax.axvline(spk_time, color="red", alpha=0.2, linewidth=0.8)

    if windows:
        for start, end in windows:
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            ax.axvspan(start, end, color="grey", alpha=0.05)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal / offsets")
    ax.set_title("Inference comparison within PGAS window")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-fs",
        type=float,
        default=30.0,
        help="Sampling rate (Hz) for the reference ground-truth smoothing grid.",
    )
    parser.add_argument(
        "--downsample-target",
        type=float,
        default=30.0,
        help="Frame rate (Hz) to use when creating smoothed/downsamped copies for PGAS and CASCADE.",
    )
    parser.add_argument(
        "--pgas-use-cache",
        action="store_true",
        help="Reuse existing PGAS trajectories instead of re-running inference.",
    )
    parser.add_argument(
        "--skip-pgas",
        action="store_true",
        help="Skip running PGAS (useful while debugging ENS2/CASCADE paths).",
    )
    parser.add_argument(
        "--skip-ens2",
        action="store_true",
        help="Skip running ENS2 inference.",
    )
    parser.add_argument(
        "--skip-cascade",
        action="store_true",
        help="Skip running CASCADE inference.",
    )
    parser.add_argument(
        "--neuron-type",
        type=str,
        default="Exc",
        help="Which ENS2 checkpoint to load (Exc or Inh).",
    )
    parser.add_argument(
        "--plot-comparison",
        action="store_true",
        help="Display a comparison plot of raw data and inference traces within the PGAS window.",
    )
    parser.add_argument(
        "--bm-sigma-spike-gap",
        type=float,
        default=0.15,
        help="Exclude ±gap seconds around spikes when estimating the PGAS bm_sigma (set 0 to disable).",
    )
    args = parser.parse_args(argv)

    data_dir = Path("data/janelia_8f/excitatory")
    data_files = sorted(data_dir.glob("*.mat"))
    if not data_files:
        raise FileNotFoundError(f"No .mat files found under {data_dir}.")
    dataset_path = data_files[0]

    print(f"Loading dataset: {dataset_path}")
    time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    trials_native = extract_trials(time_stamps, dff)
    trials_mean = [mean_downsample_trace(trial.times, trial.values, args.downsample_target) for trial in trials_native]
    raw_time_flat, raw_trace_flat = flatten_trials(trials_native)
    down_time_flat, down_trace_flat = flatten_trials(trials_mean)
    raw_fs = 1.0 / np.median(np.diff(raw_time_flat))
    downsample_label = f"{args.downsample_target:.2f}" if np.isfinite(args.downsample_target) else "raw"
    label_token = format_tag_token(downsample_label)
    pgas_resample_token = format_tag_token(f"{PGAS_RESAMPLE_FS:g}")
    cascade_resample_token = format_tag_token(f"{CASCADE_RESAMPLE_FS:g}")

    global_start = min(trial.times[0] for trial in trials_native)
    global_end = max(trial.times[-1] for trial in trials_native)
    ref_time, ref_trace = build_ground_truth_series(
        spike_times, global_start, global_end, reference_fs=args.reference_fs
    )

    methods: List[MethodResult] = []
    dataset_tag = dataset_path.stem
    pgas_windows: Optional[List[Tuple[float, float]]] = None

    edges_path = Path("results/excitatory_time_stamp_edges.npy")
    if not edges_path.exists():
        raise FileNotFoundError(
            f"PGAS window file not found at {edges_path}. Please generate it before running."
        )
    edges_lookup = np.load(edges_path, allow_pickle=True).item()
    if dataset_tag not in edges_lookup:
        raise KeyError(f"No PGAS window entry for dataset '{dataset_tag}' in {edges_path}.")
    dataset_edges = np.asarray(edges_lookup[dataset_tag], dtype=np.float64)
    trials_for_pgas = trim_trials_by_edges(trials_mean, dataset_edges)
    trials_for_pgas = resample_trials_to_fs(trials_for_pgas, PGAS_RESAMPLE_FS)
    pgas_input_time_flat, pgas_input_trace_flat = flatten_trials(trials_for_pgas)
    sigma_series: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if trials_mean:
        resampled_full = resample_trials_to_fs(trials_mean, PGAS_RESAMPLE_FS)
        sigma_time_flat, sigma_trace_flat = flatten_trials(resampled_full)
        mask = build_low_activity_mask(sigma_time_flat, spike_times, args.bm_sigma_spike_gap)
        if np.count_nonzero(mask) >= 2:
            sigma_series = (sigma_time_flat[mask], sigma_trace_flat[mask])
    pgas_trace_hash = hash_series(pgas_input_time_flat, pgas_input_trace_flat)
    pgas_output_root = Path("results/pgas_output/comparison")
    base_constants = Path("parameter_files/constants_GCaMP8_soma.json")
    pgas_gparam = Path("src/c_spikes/pgas/20230525_gold.dat")
    pgas_burnin = 100
    pgas_niter = 200
    maxspikes = maxspikes_for_rate(args.downsample_target, raw_fs)
    if sigma_series is not None:
        bm_sigma = derive_bm_sigma(
            sigma_series[0],
            sigma_series[1],
            target_fs=PGAS_RESAMPLE_FS,
        )
    else:
        bm_sigma = derive_bm_sigma(
            pgas_input_time_flat,
            pgas_input_trace_flat,
            target_fs=PGAS_RESAMPLE_FS,
        )
    bm_token = format_tag_token(f"{bm_sigma:.3g}")
    constants_path = prepare_constants_with_params(
        base_constants,
        maxspikes=maxspikes,
        bm_sigma=bm_sigma,
    )
    pgas_config = {
        "niter": pgas_niter,
        "burnin": pgas_burnin,
        "downsample_target": args.downsample_target,
        "constants_file": str(constants_path),
        "gparam_file": str(pgas_gparam),
        "edge_hash": hash_array(dataset_edges),
        "maxspikes": maxspikes,
        "input_resample_fs": PGAS_RESAMPLE_FS,
        "bm_sigma": bm_sigma,
    }
    pgas_cache_tag = (
        f"{dataset_tag}_s{label_token}_ms{maxspikes}_rs{pgas_resample_token}_bm{bm_token}"
    )

    pgas_result: Optional[MethodResult] = None
    if args.skip_pgas or args.pgas_use_cache:
        cached_pgas = load_method_cache("pgas", pgas_cache_tag, pgas_config, pgas_trace_hash)
        if cached_pgas:
            print("Loaded PGAS result from cache.")
            pgas_result = cached_pgas
            pgas_result.metadata.setdefault("input_resample_fs", PGAS_RESAMPLE_FS)
            pgas_result.metadata.setdefault("maxspikes_per_bin", PGAS_MAX_SPIKES_PER_BIN)
        elif args.skip_pgas:
            raise FileNotFoundError("PGAS cache not found; run without --skip-pgas first.")
    if pgas_result is None:
        print("Running PGAS inference...")
        pgas_result = run_pgas_inference(
            trials=trials_for_pgas,
            dataset_tag=pgas_cache_tag,
            output_root=pgas_output_root,
            constants_file=constants_path,
            gparam_file=pgas_gparam,
            recompute=not args.pgas_use_cache,
        )
        pgas_result.metadata.setdefault("niter", pgas_niter)
        save_method_cache("pgas", pgas_cache_tag, pgas_result, pgas_config, pgas_trace_hash)

    pgas_result.metadata.setdefault("burnin", pgas_burnin)
    pgas_result.metadata["window_edges"] = dataset_edges.tolist()
    pgas_result.metadata.setdefault("config", ensure_serializable(pgas_config))
    pgas_result.metadata["maxspikes"] = maxspikes
    pgas_result.metadata["maxspikes_per_bin"] = PGAS_MAX_SPIKES_PER_BIN
    pgas_result.metadata["input_resample_fs"] = PGAS_RESAMPLE_FS
    pgas_result.metadata["bm_sigma"] = bm_sigma
    methods.append(pgas_result)
    pgas_windows = [
        (pgas_result.time_stamps[seg.start], pgas_result.time_stamps[seg.stop - 1])
        for seg in segment_indices(pgas_result.time_stamps, pgas_result.sampling_rate)
        if seg.stop - seg.start > 0
    ]

    if not args.skip_ens2:
        ens2_config = {
            "neuron_type": args.neuron_type,
            "downsample_target": args.downsample_target,
            "pretrained_dir": str(Path("results/Pretrained_models/ens2_published").resolve()),
        }
        ens2_trace_hash = hash_series(raw_time_flat, raw_trace_flat)
        ens2_cached = load_method_cache("ens2", dataset_tag, ens2_config, ens2_trace_hash)
        if ens2_cached:
            print("Loaded ENS2 result from cache.")
            ens2_result = ens2_cached
        else:
            print("Running ENS2 inference...")
            ens2_result = run_ens2_inference(
                raw_time_stamps=time_stamps,
                raw_traces=dff,
                dataset_tag=dataset_tag,
                pretrained_dir=Path("results/Pretrained_models/ens2_published"),
                neuron_type=args.neuron_type,
            )
            save_method_cache("ens2", dataset_tag, ens2_result, ens2_config, ens2_trace_hash)
        ens2_result.metadata.setdefault("config", ensure_serializable(ens2_config))
        methods.append(ens2_result)

    cascade_trials = resample_trials_to_fs(trials_mean, CASCADE_RESAMPLE_FS)
    cascade_time_flat, cascade_trace_flat = flatten_trials(cascade_trials)

    if not args.skip_cascade:
        cascade_config = {
            "model_name": "Cascade_Universal_30Hz",
            "downsample_target": args.downsample_target,
            "input_resample_fs": CASCADE_RESAMPLE_FS,
        }
        cascade_cache_tag = f"{dataset_tag}_s{label_token}_rs{cascade_resample_token}"
        cascade_trace_hash = hash_series(cascade_time_flat, cascade_trace_flat)
        cascade_cached = load_method_cache("cascade", cascade_cache_tag, cascade_config, cascade_trace_hash)
        if cascade_cached:
            print("Loaded CASCADE result from cache.")
            cascade_result = cascade_cached
            cascade_result.metadata.setdefault("input_resample_fs", CASCADE_RESAMPLE_FS)
        else:
            print("Running CASCADE inference...")
            cascade_result = run_cascade_inference(
                trials=cascade_trials,
                dataset_tag=cascade_cache_tag,
                model_folder=Path("results/Pretrained_models"),
            )
            save_method_cache("cascade", cascade_cache_tag, cascade_result, cascade_config, cascade_trace_hash)
        cascade_result.metadata.setdefault("config", ensure_serializable(cascade_config))
        cascade_result.metadata["input_resample_fs"] = CASCADE_RESAMPLE_FS
        methods.append(cascade_result)

    if not methods:
        print("No methods selected; exiting.")
        return

    print("\nComputing smoothed correlations (50 ms kernel)...")
    correlations = compute_correlations(methods, ref_time, ref_trace, windows=pgas_windows)
    for method_name, corr in correlations.items():
        print(f"  {method_name:8s}: {corr if np.isfinite(corr) else 'nan'}")

    if args.plot_comparison:
        plot_inference_overlay(
            raw_time=raw_time_flat,
            raw_trace=raw_trace_flat,
            down_time=down_time_flat,
            down_trace=down_trace_flat,
            methods=methods,
            spike_times=spike_times,
            windows=pgas_windows,
            pgas_time=pgas_result.time_stamps,
            pgas_reconstruction=pgas_result.reconstruction,
            reference_time=ref_time,
            reference_trace=ref_trace,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
