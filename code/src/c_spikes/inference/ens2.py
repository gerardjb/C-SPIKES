from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .cache import load_method_cache, save_method_cache
from .types import MethodResult, ensure_serializable, hash_series


@dataclass
class Ens2Config:
    dataset_tag: str
    pretrained_dir: Path
    neuron_type: str = "Exc"
    downsample_label: str = "raw"
    use_cache: bool = True


def _segment_indices(times: np.ndarray, fs_est: float, gap_factor: float = 4.0) -> list[slice]:
    times = np.asarray(times, dtype=np.float64).ravel()
    if times.size == 0:
        return []
    if not np.isfinite(fs_est) or fs_est <= 0:
        raise ValueError("fs_est must be positive and finite when segmenting.")
    diffs = np.diff(times)
    base_dt = 1.0 / fs_est
    threshold = gap_factor * base_dt
    breaks = np.where((diffs > threshold) | ~np.isfinite(diffs))[0] + 1
    indices = np.concatenate([[0], breaks, [times.size]])
    segments: list[slice] = []
    for start, end in zip(indices[:-1], indices[1:]):
        if end > start:
            segments.append(slice(int(start), int(end)))
    return segments


def _resample_time_axis(source_times: np.ndarray, target_len: int) -> np.ndarray:
    source_times = np.asarray(source_times, dtype=np.float64).ravel()
    if target_len <= 0:
        return np.asarray([], dtype=np.float64)
    if source_times.size == 0:
        return np.full(target_len, np.nan, dtype=np.float64)
    if source_times.size == 1:
        return np.full(target_len, float(source_times[0]), dtype=np.float64)

    src_pos = np.linspace(0.0, 1.0, source_times.size, dtype=np.float64)
    dst_pos = np.linspace(0.0, 1.0, int(target_len), dtype=np.float64)
    return np.interp(dst_pos, src_pos, source_times)


def _remap_result_timebase_to_raw(
    result: MethodResult,
    raw_time_stamps: np.ndarray,
    *,
    valid_lengths: Optional[Sequence[int]] = None,
) -> MethodResult:
    from .types import TrialSeries, compute_sampling_rate, flatten_trials

    n_trials = int(raw_time_stamps.shape[0])
    fs_est = float(result.sampling_rate) if np.isfinite(result.sampling_rate) and result.sampling_rate > 0 else None
    if fs_est is None:
        fs_est = compute_sampling_rate(result.time_stamps)
    segments = _segment_indices(result.time_stamps, fs_est)
    if len(segments) != n_trials:
        return result

    time_segments: list[np.ndarray] = []
    rate_segments: list[np.ndarray] = []
    discrete_segments: list[np.ndarray] = []

    for trial_idx, seg in enumerate(segments):
        seg_rate = np.asarray(result.spike_prob[seg], dtype=np.float64).ravel()
        seg_discrete = (
            np.asarray(result.discrete_spikes[seg], dtype=np.float64).ravel()
            if result.discrete_spikes is not None
            else np.full(seg_rate.shape, np.nan, dtype=np.float64)
        )

        trial_times = np.asarray(raw_time_stamps[trial_idx], dtype=np.float64).ravel()
        valid_len = (
            int(valid_lengths[trial_idx]) if valid_lengths is not None and trial_idx < len(valid_lengths) else None
        )
        if valid_len is not None:
            if valid_len < 2:
                seg_times = np.full(seg_rate.shape, np.nan, dtype=np.float64)
            else:
                seg_times = _resample_time_axis(trial_times, seg_rate.size)
                valid_end = float(trial_times[valid_len - 1])
                invalid = seg_times > valid_end
                seg_rate = seg_rate.copy()
                seg_discrete = seg_discrete.copy()
                seg_rate[invalid] = np.nan
                seg_discrete[invalid] = np.nan
        else:
            seg_times = _resample_time_axis(trial_times, seg_rate.size)

        time_segments.append(seg_times)
        rate_segments.append(seg_rate)
        discrete_segments.append(seg_discrete)

    times, rates = flatten_trials(
        [TrialSeries(times=t, values=v) for t, v in zip(time_segments, rate_segments)]
    )
    _, discrete = flatten_trials(
        [TrialSeries(times=t, values=v) for t, v in zip(time_segments, discrete_segments)]
    )
    new_fs = compute_sampling_rate(times)
    metadata = dict(result.metadata or {})
    metadata["timebase"] = "raw_time_stamps_interp"
    return MethodResult(
        name=result.name,
        time_stamps=times,
        spike_prob=rates,
        sampling_rate=new_fs,
        metadata=metadata,
        reconstruction=result.reconstruction,
        discrete_spikes=discrete,
    )


def run_ens2_inference(
    raw_time_stamps: np.ndarray,
    raw_traces: np.ndarray,
    config: Ens2Config,
    valid_lengths: Optional[Sequence[int]] = None,
) -> MethodResult:
    trace_hash = hash_series(raw_time_stamps.ravel(), raw_traces.ravel())
    cfg_dict: Dict[str, Any] = {
        "neuron_type": config.neuron_type,
        "pretrained_dir": str(config.pretrained_dir),
        "downsample_target": config.downsample_label,
    }
    if config.use_cache:
        cached = load_method_cache("ens2", config.dataset_tag, cfg_dict, trace_hash)
        if cached:
            remapped = _remap_result_timebase_to_raw(cached, raw_time_stamps, valid_lengths=valid_lengths)
            if remapped is cached:
                return cached
            # Best-effort persist the corrected timebase so subsequent runs don't repeatedly remap.
            # (May fail in read-only/shared environments.)
            try:
                save_method_cache("ens2", config.dataset_tag, remapped, cfg_dict, trace_hash)
            except OSError:
                pass
            return remapped

    try:
        ens2_module = __import__("c_spikes.ens2.ENS2", fromlist=["ENS2", "compile_test_data"])
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ENS2 module not found. Ensure the upstream ENS2 package is available on PYTHONPATH."
        ) from exc

    import torch

    try:
        from torch.serialization import add_safe_globals  # type: ignore
    except ImportError:  # pragma: no cover
        add_safe_globals = None  # type: ignore
    try:
        from tqdm.auto import trange
    except ImportError:  # pragma: no cover

        def trange(n, **kwargs):
            return range(n)

    sys_modules = __import__("sys").modules
    sys_modules.setdefault("ENS2", ens2_module)
    ens2 = ens2_module.ENS2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ens2.DEVICE = device

    pretrained_dir = config.pretrained_dir.expanduser()
    if config.neuron_type.lower().startswith("exc"):
        state_dict_path = pretrained_dir / "exc_ens2_pub.pt"
    else:
        state_dict_path = pretrained_dir / "inh_ens2_pub.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"ENS2 checkpoint not found: {state_dict_path}")

    if add_safe_globals and hasattr(ens2_module, "UNet"):
        try:
            add_safe_globals([ens2_module.UNet])
        except Exception:
            pass

    load_kwargs = {"map_location": torch.device(device)}
    try:
        checkpoint = torch.load(state_dict_path, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(state_dict_path, **load_kwargs)

    state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    trial_duration = float(raw_time_stamps[0, -1] - raw_time_stamps[0, 0])
    test_data = ens2_module.compile_test_data(raw_traces, trial_duration)

    time_segments = []
    rate_segments = []
    discrete_segments = []
    event_counts = []
    frame_rates = []

    for trial_idx in trange(len(test_data), desc="ENS2 inference", leave=False):
        dff_segment = test_data[trial_idx]["dff_resampled_segment"]
        _, temp_pd_rate, temp_pd_spike, temp_pd_event = ens2.predict(dff_segment, state_dict=state_dict)

        trial_times = np.asarray(raw_time_stamps[trial_idx], dtype=np.float64).ravel()
        rate_values = np.asarray(temp_pd_rate, dtype=np.float64).ravel()
        discrete_values = np.asarray(temp_pd_spike, dtype=np.float64).ravel()

        abs_time = _resample_time_axis(trial_times, rate_values.size)
        valid_len = int(valid_lengths[trial_idx]) if valid_lengths is not None and trial_idx < len(valid_lengths) else None
        if valid_len is not None and valid_len >= 2:
            valid_end = float(trial_times[valid_len - 1])
            invalid = abs_time > valid_end
            if invalid.any():
                rate_values = rate_values.copy()
                discrete_values = discrete_values.copy()
                rate_values[invalid] = np.nan
                discrete_values[invalid] = np.nan

        time_segments.append(abs_time)
        rate_segments.append(rate_values)
        discrete_segments.append(discrete_values)
        event_counts.append(len(temp_pd_event))
        frame_rates.append(float(test_data[trial_idx]["frame_rate_resampled"]))

    from .types import TrialSeries, flatten_trials

    times, rates = flatten_trials(
        [TrialSeries(times=t, values=s) for t, s in zip(time_segments, rate_segments)]
    )
    _, discrete = flatten_trials(
        [TrialSeries(times=t, values=s) for t, s in zip(time_segments, discrete_segments)]
    )
    from .types import compute_sampling_rate

    fs_est = compute_sampling_rate(times)

    result = MethodResult(
        name="ens2",
        time_stamps=times,
        spike_prob=rates,
        sampling_rate=fs_est,
        metadata={
            "device": device,
            "checkpoint": str(state_dict_path),
            "neuron_type": config.neuron_type,
            "event_counts": event_counts,
            "frame_rates": frame_rates,
            "timebase": "raw_time_stamps_interp",
            "config": ensure_serializable(cfg_dict),
        },
        discrete_spikes=discrete,
    )
    try:
        save_method_cache("ens2", config.dataset_tag, result, cfg_dict, trace_hash)
    except OSError:
        pass
    return result
