from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cache import load_method_cache, save_method_cache
from .eval import segment_indices
from .smoothing import resample_trials_to_fs
from .types import (
    MethodResult,
    TrialSeries,
    compute_sampling_rate,
    ensure_serializable,
    hash_array,
    hash_series,
)


PGAS_RESAMPLE_FS: float = 120.0
PGAS_MAX_SPIKES_PER_BIN: int = 1
PGAS_BURNIN: int = 100
PGAS_NITER: int = 200


@dataclass
class PgasConfig:
    dataset_tag: str
    output_root: Path
    constants_file: Path
    gparam_file: Path
    resample_fs: float = PGAS_RESAMPLE_FS
    niter: int = PGAS_NITER
    burnin: int = PGAS_BURNIN
    downsample_label: str = "raw"
    maxspikes: Optional[int] = None
    maxspikes_per_bin: int = PGAS_MAX_SPIKES_PER_BIN
    bm_sigma: Optional[float] = None
    bm_sigma_gap_s: float = 0.15
    edges: Optional[np.ndarray] = None
    use_cache: bool = True


def maxspikes_for_rate(target_fs: Optional[float], native_fs: float) -> int:
    if target_fs is None or np.isclose(target_fs, native_fs):
        return PGAS_MAX_SPIKES_PER_BIN + 1
    if target_fs <= 0:
        raise ValueError("target_fs must be positive when provided.")

    ratio = max(native_fs / target_fs, 1.0)
    dynamic_limit = max(PGAS_MAX_SPIKES_PER_BIN + 2, int(np.ceil(ratio)) + 1)

    if np.isclose(target_fs, 30.0, atol=1e-1):
        return max(4, int(np.ceil(ratio * 0.5)) + 1)
    if np.isclose(target_fs, 10.0, atol=1e-1):
        return max(9, dynamic_limit)
    return dynamic_limit


def build_low_activity_mask(
    sample_times: np.ndarray,
    spike_times: np.ndarray,
    exclusion: float,
) -> np.ndarray:
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


def compute_robust_diff_std(
    times: np.ndarray,
    values: np.ndarray,
    clip_percentiles: Tuple[float, float] = (5.0, 95.0),
) -> float:
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


def derive_bm_sigma(
    times: np.ndarray,
    values: np.ndarray,
    target_fs: float,
    scale_factor: float = 0.25,
    min_sigma: float = 5e-4,
    max_sigma: float = 5e-2,
) -> float:
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")
    diff_std = compute_robust_diff_std(times, values)
    if diff_std <= 0:
        return float(min_sigma)
    dt = 1.0 / target_fs
    bm_sigma = scale_factor * diff_std / np.sqrt(dt)
    return float(np.clip(bm_sigma, min_sigma, max_sigma))


def build_constants_cache_path(base_constants: Path, tokens: Sequence[str]) -> Path:
    from .types import ensure_serializable  # unused import hint

    cache_dir = Path("results") / "inference_cache" / "pgas_constants"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(tokens)
    return cache_dir / f"{base_constants.stem}_{suffix}{base_constants.suffix}"


def format_tag_token(value: str) -> str:
    return value.replace(" ", "_").replace(".", "p")


def prepare_constants_with_params(
    base_constants: Path,
    *,
    maxspikes: int,
    bm_sigma: Optional[float] = None,
) -> Path:
    base_constants = Path(base_constants)
    tokens = [f"ms{maxspikes}"]
    if bm_sigma is not None:
        tokens.append(f"bm{format_tag_token(f'{bm_sigma:.4g}')}")
    target_path = build_constants_cache_path(base_constants, tokens)
    if target_path.exists():
        return target_path
    import json

    with base_constants.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("MCMC", {})["maxspikes"] = int(maxspikes)
    if bm_sigma is not None:
        data.setdefault("BM", {})["bm_sigma"] = float(bm_sigma)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return target_path


def estimate_bm_sigma_for_trials(
    trials: Sequence[TrialSeries],
    spike_times: np.ndarray,
    resample_fs: float,
    gap_s: float,
) -> float:
    resampled = resample_trials_to_fs(trials, resample_fs)
    from .types import flatten_trials

    sigma_time_flat, sigma_trace_flat = flatten_trials(resampled)
    mask = build_low_activity_mask(sigma_time_flat, spike_times, gap_s)
    if np.count_nonzero(mask) >= 2:
        sigma_times = sigma_time_flat[mask]
        sigma_values = sigma_trace_flat[mask]
    else:
        sigma_times = sigma_time_flat
        sigma_values = sigma_trace_flat
    return derive_bm_sigma(sigma_times, sigma_values, target_fs=resample_fs)


def run_pgas_inference(
    trials: Sequence[TrialSeries],
    raw_fs: float,
    spike_times: np.ndarray,
    config: PgasConfig,
) -> MethodResult:
    trials_for_pgas: Sequence[TrialSeries]
    if config.edges is not None:
        from compare_inference_methods import trim_trials_by_edges  # reuse existing helper

        trials_for_pgas = trim_trials_by_edges(trials, config.edges)
    else:
        trials_for_pgas = list(trials)

    trials_resampled = resample_trials_to_fs(trials_for_pgas, config.resample_fs)
    from .types import flatten_trials

    time_flat, trace_flat = flatten_trials(trials_resampled)
    trace_hash = hash_series(time_flat, trace_flat)
    maxspikes = config.maxspikes if config.maxspikes is not None else maxspikes_for_rate(
        config.resample_fs, raw_fs
    )
    bm_sigma = (
        config.bm_sigma
        if config.bm_sigma is not None
        else estimate_bm_sigma_for_trials(trials_for_pgas, spike_times, config.resample_fs, config.bm_sigma_gap_s)
    )
    constants_path = prepare_constants_with_params(
        config.constants_file,
        maxspikes=maxspikes,
        bm_sigma=bm_sigma,
    )
    label_token = format_tag_token(config.downsample_label)
    pgas_resample_token = format_tag_token(f"{config.resample_fs:g}")
    bm_token = format_tag_token(f"{bm_sigma:.3g}")
    run_tag = f"{config.dataset_tag}_s{label_token}_ms{maxspikes}_rs{pgas_resample_token}_bm{bm_token}"
    cfg_dict = {
        "niter": config.niter,
        "burnin": config.burnin,
        "downsample_target": config.downsample_label,
        "constants_file": str(constants_path),
        "gparam_file": str(config.gparam_file),
        "maxspikes": maxspikes,
        "input_resample_fs": config.resample_fs,
        "bm_sigma": bm_sigma,
    }
    if config.edges is not None:
        cfg_dict["edge_hash"] = hash_array(config.edges)

    if config.use_cache:
        cached = load_method_cache("pgas", run_tag, cfg_dict, trace_hash)
        if cached:
            cached.metadata.setdefault("input_resample_fs", config.resample_fs)
            cached.metadata.setdefault("maxspikes_per_bin", config.maxspikes_per_bin)
            cached.metadata.setdefault("cache_tag", run_tag)
            cached.metadata.setdefault("maxspikes", maxspikes)
            cached.metadata.setdefault("bm_sigma", bm_sigma)
            return cached

    try:
        pgas_mod = __import__("c_spikes.pgas.pgas_bound", fromlist=["Analyzer"])
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PGAS module not found. Build the c_spikes.pgas extension before running."
        ) from exc

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    trial_list = list(trials_resampled)
    for trial_idx, trial in enumerate(trial_list):
        tag = f"{run_tag}_trial{trial_idx}"
        analyzer = pgas_mod.Analyzer(
            time=np.ascontiguousarray(trial.times, dtype=np.float64),
            data=np.ascontiguousarray(trial.values, dtype=np.float64),
            constants_file=str(constants_path),
            output_folder=str(output_root),
            column=1,
            tag=tag,
            niter=config.niter,
            trainedPriorFile="",
            append=False,
            trim=1,
            verbose=False,
            gtSpikes=np.zeros(0, dtype=np.float64),
            has_trained_priors=False,
            has_gtspikes=False,
            maxlen=int(trial.values.size),
            Gparam_file=str(config.gparam_file),
            seed=2 + trial_idx,
        )
        analyzer.run()

    from compare_inference_methods import load_pgas_method_result  # reuse existing loader

    traces = load_pgas_method_result(
        trials=trial_list,
        dataset_tag=run_tag,
        output_root=output_root,
        burnin=config.burnin,
        metadata={
            "burnin": config.burnin,
            "niter": config.niter,
            "output_root": str(output_root),
        },
    )
    traces.metadata.setdefault("config", ensure_serializable(cfg_dict))
    traces.metadata.setdefault("maxspikes", maxspikes)
    traces.metadata.setdefault("maxspikes_per_bin", config.maxspikes_per_bin)
    traces.metadata.setdefault("input_resample_fs", config.resample_fs)
    traces.metadata.setdefault("bm_sigma", bm_sigma)
    traces.metadata.setdefault("cache_tag", run_tag)
    save_method_cache("pgas", run_tag, traces, cfg_dict, trace_hash)
    return traces


def pgas_windows_from_result(result: MethodResult) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    for seg in segment_indices(result.time_stamps, result.sampling_rate):
        if seg.stop - seg.start <= 0:
            continue
        start = result.time_stamps[seg.start]
        end = result.time_stamps[seg.stop - 1]
        windows.append((float(start), float(end)))
    return windows


