from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .cache import load_method_cache, save_method_cache
from .smoothing import resample_trials_to_fs
from .types import MethodResult, TrialSeries, ensure_serializable, hash_series


CASCADE_RESAMPLE_FS: float = 30.0


@dataclass
class CascadeConfig:
    dataset_tag: str
    model_folder: Path
    model_name: str = "Cascade_Universal_30Hz"
    resample_fs: float = CASCADE_RESAMPLE_FS
    downsample_label: str = "raw"
    use_cache: bool = True


def run_cascade_inference(
    trials: Sequence[TrialSeries],
    config: CascadeConfig,
) -> MethodResult:
    trials_resampled = resample_trials_to_fs(trials, config.resample_fs)
    from .types import flatten_trials

    time_flat, trace_flat = flatten_trials(trials_resampled)
    trace_hash = hash_series(time_flat, trace_flat)
    from .pgas import format_tag_token

    label_token = format_tag_token(config.downsample_label)
    resample_token = format_tag_token(f"{config.resample_fs:g}")
    cache_tag = f"{config.dataset_tag}_s{label_token}_rs{resample_token}"
    cfg_dict = {
        "model_name": config.model_name,
        "downsample_target": config.downsample_label,
        "input_resample_fs": config.resample_fs,
    }

    if config.use_cache:
        cached = load_method_cache("cascade", cache_tag, cfg_dict, trace_hash)
        if cached:
            cached.metadata.setdefault("input_resample_fs", config.resample_fs)
            cached.metadata.setdefault("cache_tag", cache_tag)
            return cached

    from c_spikes.cascade2p import cascade
    from c_spikes.cascade2p.utils_discrete_spikes import infer_discrete_spikes

    min_len = min(trial.values.size for trial in trials_resampled)
    if any(trial.values.size != min_len for trial in trials_resampled):
        aligned_trials = [
            TrialSeries(times=trial.times[:min_len], values=trial.values[:min_len])
            for trial in trials_resampled
        ]
    else:
        aligned_trials = list(trials_resampled)

    traces_matrix = np.stack([trial.values for trial in aligned_trials], axis=0)
    time_matrix = np.stack([trial.times for trial in aligned_trials], axis=0)
    frame_rate = float(1.0 / np.median(np.diff(time_matrix[0])))

    pred_rate = cascade.predict(
        config.model_name,
        traces_matrix,
        model_folder=str(config.model_folder),
        reuse_models=True,
    )

    approximations, spike_lists = infer_discrete_spikes(
        pred_rate, config.model_name, model_folder=str(config.model_folder)
    )
    discrete_matrix = np.zeros_like(pred_rate, dtype=float)
    for neuron_idx, spike_list in enumerate(spike_lists):
        if spike_list is None:
            continue
        indices = np.asarray(spike_list, dtype=int)
        indices = indices[(indices >= 0) & (indices < discrete_matrix.shape[1])]
        discrete_matrix[neuron_idx, indices] += 1

    from .types import flatten_trials, compute_sampling_rate

    time_flat_arr, rate_flat_arr = flatten_trials(
        [TrialSeries(times=time_matrix[idx], values=pred_rate[idx]) for idx in range(pred_rate.shape[0])]
    )
    _, discrete_flat_arr = flatten_trials(
        [TrialSeries(times=time_matrix[idx], values=discrete_matrix[idx]) for idx in range(discrete_matrix.shape[0])]
    )
    fs_est = compute_sampling_rate(time_flat_arr)
    result = MethodResult(
        name="cascade",
        time_stamps=time_flat_arr,
        spike_prob=rate_flat_arr,
        sampling_rate=fs_est,
        metadata={
            "input_resample_fs": config.resample_fs,
            "cache_tag": cache_tag,
            "config": ensure_serializable(cfg_dict),
        },
        discrete_spikes=discrete_flat_arr,
    )
    save_method_cache("cascade", cache_tag, result, cfg_dict, trace_hash)
    return result
