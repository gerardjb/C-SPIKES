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
            return cached

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

    trial_starts = raw_time_stamps[:, 0]
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

        fluo_times = test_data[trial_idx]["fluo_times_resampled"]
        abs_time = np.asarray(fluo_times + trial_starts[trial_idx], dtype=np.float64).ravel()
        rate_values = np.asarray(temp_pd_rate, dtype=np.float64).ravel()
        discrete_values = np.asarray(temp_pd_spike, dtype=np.float64).ravel()

        valid_len = (
            int(valid_lengths[trial_idx]) if valid_lengths is not None and trial_idx < len(valid_lengths) else None
        )
        if valid_len is not None and valid_len < rate_values.size:
            rate_values[valid_len:] = np.nan
            discrete_values[valid_len:] = np.nan

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
            "config": ensure_serializable(cfg_dict),
        },
        discrete_spikes=discrete,
    )
    save_method_cache("ens2", config.dataset_tag, result, cfg_dict, trace_hash)
    return result

