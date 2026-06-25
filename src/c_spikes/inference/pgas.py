from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import re

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
    flatten_trials,
)
from c_spikes.utils import unroll_mean_pgas_traj


PGAS_RESAMPLE_FS: float = 120.0
PGAS_MAX_SPIKES_PER_BIN: int = 1
PGAS_BURNIN: int = 100
PGAS_NITER: int = 200
PGAS_BM_SIGMA_DEFAULT: float = 2e-2
PGAS_BM_SIGMA_MIN: float = 5e-4
PGAS_BM_SIGMA_MAX: float = 5e-1
PGAS_SIGMA2_TARGET_MIN: float = 5e-6
PGAS_SIGMA2_TARGET_MAX: float = 8e-2
PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT: float = 4.0
PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT: str = "inference"
PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT: str = "dataset"
PGAS_NOISE_CALIBRATION_SCOPES: Tuple[str, ...] = ("inference", "full")
PGAS_NOISE_CALIBRATION_GRANULARITIES: Tuple[str, ...] = ("dataset", "trial")


@dataclass
class PgasConfig:
    dataset_tag: str
    output_root: Path
    constants_file: Path
    gparam_file: Path
    resample_fs: Optional[float] = None  # None => use raw/native rate
    niter: int = PGAS_NITER
    burnin: int = PGAS_BURNIN
    downsample_label: str = "raw"
    maxspikes: Optional[int] = None
    maxspikes_per_bin: int = PGAS_MAX_SPIKES_PER_BIN
    bm_sigma: Optional[float] = None
    bm_sigma_min: float = PGAS_BM_SIGMA_MIN
    bm_sigma_max: float = PGAS_BM_SIGMA_MAX
    bm_sigma_gap_s: float = 0.15
    bm_sigma_use_low_activity_mask: bool = False
    sigma2_target: Optional[float] = None
    sigma2_alpha: Optional[float] = None
    sigma2_prior_strength: float = PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT
    noise_calibration_scope: str = PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT
    noise_calibration_granularity: str = PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT
    edges: Optional[np.ndarray] = None
    use_cache: bool = True


def validate_bm_sigma_bounds(min_sigma: float, max_sigma: float) -> Tuple[float, float]:
    min_sigma = float(min_sigma)
    max_sigma = float(max_sigma)
    if not np.isfinite(min_sigma) or min_sigma <= 0:
        raise ValueError(f"bm_sigma minimum must be positive and finite; got {min_sigma!r}.")
    if not np.isfinite(max_sigma) or max_sigma <= 0:
        raise ValueError(f"bm_sigma maximum must be positive and finite; got {max_sigma!r}.")
    if max_sigma < min_sigma:
        raise ValueError(
            f"bm_sigma maximum must be >= minimum; got min={min_sigma:g}, max={max_sigma:g}."
        )
    return min_sigma, max_sigma


def normalize_noise_calibration_scope(scope: str) -> str:
    token = str(scope).strip().lower()
    if token not in PGAS_NOISE_CALIBRATION_SCOPES:
        raise ValueError(
            "noise_calibration_scope must be one of "
            f"{PGAS_NOISE_CALIBRATION_SCOPES}; got {scope!r}."
        )
    return token


def normalize_noise_calibration_granularity(granularity: str) -> str:
    token = str(granularity).strip().lower()
    if token not in PGAS_NOISE_CALIBRATION_GRANULARITIES:
        raise ValueError(
            "noise_calibration_granularity must be one of "
            f"{PGAS_NOISE_CALIBRATION_GRANULARITIES}; got {granularity!r}."
        )
    return token


@dataclass
class PgasNoiseCalibration:
    bm_sigma: float
    sigma2_target: float
    diff_var: float
    diff2_var: float
    qdt: float
    n_samples: int
    used_low_activity_mask: bool


@dataclass
class PgasNoiseSettings:
    bm_sigma: float
    sigma2_target: Optional[float]
    sigma2_alpha: Optional[float]
    sigma2_beta: Optional[float]
    sigma2_prior_strength: float
    constants_path: Path
    calibration: Optional[PgasNoiseCalibration]
    noise_calibration: Optional[Dict[str, object]]
    cfg: Dict[str, object]


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
        # Historically we used a much smaller cap at 10Hz (100ms bins) to keep the
        # PGAS state space tractable; this also matches existing cached runs under
        # `results/full_evaluation_by_run/base` (ms7).
        return max(7, int(np.ceil(ratio * 0.5)))
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
    return _robust_scale(diffs, clip_percentiles=clip_percentiles)


def _robust_scale(
    values: np.ndarray,
    clip_percentiles: Optional[Tuple[float, float]] = (5.0, 95.0),
) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    if clip_percentiles is not None and vals.size >= 4:
        lo, hi = np.percentile(vals, clip_percentiles)
        keep = (vals >= lo) & (vals <= hi)
        if keep.any():
            vals = vals[keep]
    if vals.size == 0:
        return 0.0
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad <= 0:
        return float(np.std(vals)) if vals.size > 0 else 0.0
    return float(1.4826 * mad)


def derive_bm_sigma(
    times: np.ndarray,
    values: np.ndarray,
    target_fs: float,
    scale_factor: float = 0.25,
    min_sigma: float = PGAS_BM_SIGMA_MIN,
    max_sigma: float = PGAS_BM_SIGMA_MAX,
) -> float:
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")
    min_sigma, max_sigma = validate_bm_sigma_bounds(min_sigma, max_sigma)
    diff_std = compute_robust_diff_std(times, values)
    if diff_std <= 0:
        return float(min_sigma)
    dt = 1.0 / target_fs
    bm_sigma = scale_factor * diff_std / np.sqrt(dt)
    return float(np.clip(bm_sigma, min_sigma, max_sigma))


def derive_bm_sigma_and_sigma2(
    times: np.ndarray,
    values: np.ndarray,
    target_fs: float,
    *,
    min_bm_sigma: float = PGAS_BM_SIGMA_MIN,
    max_bm_sigma: float = PGAS_BM_SIGMA_MAX,
    min_sigma2_target: float = PGAS_SIGMA2_TARGET_MIN,
    max_sigma2_target: float = PGAS_SIGMA2_TARGET_MAX,
    clip_percentiles: Optional[Tuple[float, float]] = (5.0, 95.0),
    used_low_activity_mask: bool = False,
) -> PgasNoiseCalibration:
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")
    min_bm_sigma, max_bm_sigma = validate_bm_sigma_bounds(
        min_bm_sigma,
        max_bm_sigma,
    )
    times = np.asarray(times, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if times.size < 3:
        return PgasNoiseCalibration(
            bm_sigma=float(min_bm_sigma),
            sigma2_target=float(min_sigma2_target),
            diff_var=0.0,
            diff2_var=0.0,
            qdt=0.0,
            n_samples=int(times.size),
            used_low_activity_mask=bool(used_low_activity_mask),
        )
    order = np.argsort(times)
    y = values[order]
    d1 = np.diff(y)
    d2 = np.diff(d1)
    diff_std = _robust_scale(d1, clip_percentiles=clip_percentiles)
    diff2_std = _robust_scale(d2, clip_percentiles=clip_percentiles)
    diff_var = float(diff_std**2)
    diff2_var = float(diff2_std**2)
    qdt = max(3.0 * diff_var - diff2_var, 0.0)
    sigma2_target = max((diff2_var - 2.0 * diff_var) * 0.5, 0.0)
    dt = 1.0 / float(target_fs)
    if qdt <= 0 or dt <= 0:
        bm_sigma = float(min_bm_sigma)
    else:
        bm_sigma = float(np.sqrt(qdt / dt))
    bm_sigma = float(np.clip(bm_sigma, min_bm_sigma, max_bm_sigma))
    sigma2_target = float(np.clip(sigma2_target, min_sigma2_target, max_sigma2_target))
    return PgasNoiseCalibration(
        bm_sigma=bm_sigma,
        sigma2_target=sigma2_target,
        diff_var=diff_var,
        diff2_var=diff2_var,
        qdt=float(qdt),
        n_samples=int(y.size),
        used_low_activity_mask=bool(used_low_activity_mask),
    )


def build_constants_cache_path(base_constants: Path, tokens: Sequence[str]) -> Path:
    from .cache import get_cache_root
    from .types import ensure_serializable  # unused import hint

    cache_dir = get_cache_root() / "pgas_constants"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(tokens)
    return cache_dir / f"{base_constants.stem}_{suffix}{base_constants.suffix}"


def format_tag_token(value: str) -> str:
    return value.replace(" ", "_").replace(".", "p")


def normalize_sigma2_prior_strength(
    sigma2_prior_strength: float,
) -> float:
    strength = float(sigma2_prior_strength)
    if not np.isfinite(strength) or strength <= 0:
        return PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT
    return strength


def map_sigma2_target_to_ig_params(
    sigma2_target: float,
    *,
    sigma2_alpha: Optional[float] = None,
    sigma2_prior_strength: float = PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT,
) -> Tuple[float, float, float]:
    sigma2_target_clipped = float(
        np.clip(float(sigma2_target), PGAS_SIGMA2_TARGET_MIN, PGAS_SIGMA2_TARGET_MAX)
    )
    if sigma2_alpha is not None:
        alpha = float(sigma2_alpha)
    else:
        alpha = 2.0 + normalize_sigma2_prior_strength(sigma2_prior_strength)
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 2.0 + PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT
    beta = float(sigma2_target_clipped * (alpha + 1.0))
    return sigma2_target_clipped, float(alpha), beta


def prepare_constants_with_params(
    base_constants: Path,
    *,
    maxspikes: int,
    bm_sigma: Optional[float] = None,
    sigma2_target: Optional[float] = None,
    sigma2_alpha: Optional[float] = None,
    sigma2_prior_strength: float = PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT,
) -> Path:
    base_constants = Path(base_constants)
    tokens = [f"ms{maxspikes}"]
    if bm_sigma is not None:
        tokens.append(f"bm{format_tag_token(f'{bm_sigma:.4g}')}")
    if sigma2_target is not None:
        tokens.append(f"s2{format_tag_token(f'{sigma2_target:.4g}')}")
    if sigma2_alpha is not None:
        tokens.append(f"a2{format_tag_token(f'{sigma2_alpha:.4g}')}")
    elif sigma2_target is not None:
        prior_strength = normalize_sigma2_prior_strength(sigma2_prior_strength)
        tokens.append(f"p2{format_tag_token(f'{prior_strength:.4g}')}")
    target_path = build_constants_cache_path(base_constants, tokens)
    if target_path.exists():
        return target_path
    import json

    with base_constants.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("MCMC", {})["maxspikes"] = int(maxspikes)
    if bm_sigma is not None:
        data.setdefault("BM", {})["bm_sigma"] = float(bm_sigma)
    if sigma2_target is not None:
        priors = data.setdefault("priors", {})
        _, alpha, beta = map_sigma2_target_to_ig_params(
            sigma2_target,
            sigma2_alpha=sigma2_alpha,
            sigma2_prior_strength=sigma2_prior_strength,
        )
        priors["alpha sigma2"] = float(alpha)
        priors["beta sigma2"] = float(beta)
    elif sigma2_alpha is not None:
        priors = data.setdefault("priors", {})
        alpha = float(sigma2_alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 2.0 + PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT
        priors["alpha sigma2"] = float(alpha)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return target_path


def estimate_bm_sigma_for_trials(
    trials: Sequence[TrialSeries],
    spike_times: np.ndarray,
    resample_fs: float,
    gap_s: float,
    min_bm_sigma: float = PGAS_BM_SIGMA_MIN,
    max_bm_sigma: float = PGAS_BM_SIGMA_MAX,
    use_low_activity_mask: bool = False,
) -> float:
    calibration = estimate_noise_calibration_for_trials(
        trials,
        spike_times,
        resample_fs,
        gap_s,
        use_low_activity_mask=use_low_activity_mask,
        min_bm_sigma=min_bm_sigma,
        max_bm_sigma=max_bm_sigma,
    )
    return float(calibration.bm_sigma)


def estimate_noise_calibration_for_trials(
    trials: Sequence[TrialSeries],
    spike_times: np.ndarray,
    resample_fs: float,
    gap_s: float,
    *,
    use_low_activity_mask: bool = False,
    min_bm_sigma: float = PGAS_BM_SIGMA_MIN,
    max_bm_sigma: float = PGAS_BM_SIGMA_MAX,
) -> PgasNoiseCalibration:
    resampled = resample_trials_to_fs(trials, resample_fs)
    from .types import flatten_trials

    sigma_time_flat, sigma_trace_flat = flatten_trials(resampled)
    used_mask = False
    if use_low_activity_mask:
        mask = build_low_activity_mask(sigma_time_flat, spike_times, gap_s)
    else:
        mask = np.ones_like(sigma_time_flat, dtype=bool)
    if np.count_nonzero(mask) >= 3:
        sigma_times = sigma_time_flat[mask]
        sigma_values = sigma_trace_flat[mask]
        used_mask = bool(use_low_activity_mask)
    else:
        sigma_times = sigma_time_flat
        sigma_values = sigma_trace_flat
    return derive_bm_sigma_and_sigma2(
        sigma_times,
        sigma_values,
        target_fs=resample_fs,
        used_low_activity_mask=used_mask,
        min_bm_sigma=min_bm_sigma,
        max_bm_sigma=max_bm_sigma,
    )


def _noise_calibration_metadata(
    calibration: Optional[PgasNoiseCalibration],
) -> Optional[Dict[str, object]]:
    if calibration is None:
        return None
    return {
        "method": "two_timescale_robust_diff",
        "diff_var": float(calibration.diff_var),
        "diff2_var": float(calibration.diff2_var),
        "qdt": float(calibration.qdt),
        "n_samples": int(calibration.n_samples),
        "used_low_activity_mask": bool(calibration.used_low_activity_mask),
        "clip_percentiles": [5.0, 95.0],
    }


def _build_pgas_noise_settings(
    *,
    config: PgasConfig,
    maxspikes: int,
    calibration_trials: Sequence[TrialSeries],
    spike_times: np.ndarray,
    input_fs: float,
    bm_sigma_min: float,
    bm_sigma_max: float,
) -> PgasNoiseSettings:
    calibration: Optional[PgasNoiseCalibration] = None
    if config.bm_sigma is None:
        calibration = estimate_noise_calibration_for_trials(
            calibration_trials,
            spike_times,
            input_fs,
            config.bm_sigma_gap_s,
            min_bm_sigma=bm_sigma_min,
            max_bm_sigma=bm_sigma_max,
            use_low_activity_mask=config.bm_sigma_use_low_activity_mask,
        )
        bm_sigma = float(calibration.bm_sigma)
    else:
        bm_sigma = float(config.bm_sigma)

    sigma2_target = (
        float(config.sigma2_target)
        if config.sigma2_target is not None
        else (float(calibration.sigma2_target) if calibration is not None else None)
    )
    sigma2_target_effective: Optional[float] = None
    sigma2_alpha_effective: Optional[float] = None
    sigma2_beta_effective: Optional[float] = None
    if sigma2_target is not None:
        (
            sigma2_target_effective,
            sigma2_alpha_effective,
            sigma2_beta_effective,
        ) = map_sigma2_target_to_ig_params(
            sigma2_target,
            sigma2_alpha=config.sigma2_alpha,
            sigma2_prior_strength=config.sigma2_prior_strength,
        )
    elif config.sigma2_alpha is not None:
        sigma2_alpha_effective = float(config.sigma2_alpha)
        if not np.isfinite(sigma2_alpha_effective) or sigma2_alpha_effective <= 0:
            sigma2_alpha_effective = 2.0 + PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT

    sigma2_prior_strength_effective = normalize_sigma2_prior_strength(
        config.sigma2_prior_strength
    )
    constants_path = prepare_constants_with_params(
        config.constants_file,
        maxspikes=maxspikes,
        bm_sigma=bm_sigma,
        sigma2_target=sigma2_target_effective,
        sigma2_alpha=sigma2_alpha_effective if config.sigma2_alpha is not None else None,
        sigma2_prior_strength=sigma2_prior_strength_effective,
    )
    noise_calibration = _noise_calibration_metadata(calibration)

    cfg: Dict[str, object] = {
        "constants_file": str(constants_path),
        "bm_sigma": bm_sigma,
    }
    if sigma2_target_effective is not None:
        cfg["sigma2_target"] = float(sigma2_target_effective)
    if sigma2_alpha_effective is not None:
        cfg["sigma2_alpha"] = float(sigma2_alpha_effective)
    if sigma2_target_effective is not None:
        cfg["sigma2_prior_strength"] = float(sigma2_prior_strength_effective)
    if noise_calibration is not None:
        cfg["noise_calibration"] = noise_calibration

    return PgasNoiseSettings(
        bm_sigma=bm_sigma,
        sigma2_target=sigma2_target_effective,
        sigma2_alpha=sigma2_alpha_effective,
        sigma2_beta=sigma2_beta_effective,
        sigma2_prior_strength=sigma2_prior_strength_effective,
        constants_path=constants_path,
        calibration=calibration,
        noise_calibration=noise_calibration,
        cfg=cfg,
    )


def run_pgas_inference(
    trials: Sequence[TrialSeries],
    raw_fs: float,
    spike_times: np.ndarray,
    config: PgasConfig,
) -> MethodResult:
    trials_input = list(trials)
    noise_scope = normalize_noise_calibration_scope(config.noise_calibration_scope)
    noise_granularity = normalize_noise_calibration_granularity(
        config.noise_calibration_granularity
    )
    if noise_granularity == "trial" and config.bm_sigma is not None:
        raise ValueError(
            "per-trial PGAS noise calibration requires --pgas-bm-sigma=auto "
            "(PgasConfig.bm_sigma=None)."
        )

    trials_for_pgas: Sequence[TrialSeries]
    if config.edges is not None:
        trials_for_pgas = trim_trials_by_edges(trials_input, config.edges)
    else:
        trials_for_pgas = list(trials_input)

    calibration_source_trials = (
        trials_for_pgas if noise_scope == "inference" else trials_input
    )
    if len(calibration_source_trials) != len(trials_for_pgas):
        raise ValueError(
            "PGAS calibration and inference trial counts differ. "
            f"calibration={len(calibration_source_trials)}, inference={len(trials_for_pgas)}"
        )

    if config.resample_fs is not None:
        trials_resampled = resample_trials_to_fs(trials_for_pgas, config.resample_fs)
    else:
        trials_resampled = list(trials_for_pgas)
    from .types import flatten_trials

    time_flat, trace_flat = flatten_trials(trials_resampled)
    trace_hash = hash_series(time_flat, trace_flat)
    input_fs = compute_sampling_rate(time_flat)
    maxspikes = (
        config.maxspikes
        if config.maxspikes is not None
        else maxspikes_for_rate(input_fs, raw_fs)
    )
    bm_sigma_min, bm_sigma_max = validate_bm_sigma_bounds(
        config.bm_sigma_min,
        config.bm_sigma_max,
    )
    label_token = format_tag_token(config.downsample_label)
    pgas_resample_token = "raw" if config.resample_fs is None else format_tag_token(f"{config.resample_fs:g}")
    run_tag_base = (
        f"{config.dataset_tag}_s{label_token}_ms{maxspikes}_rs{pgas_resample_token}"
    )
    mode_tag_suffix = ""
    if noise_scope != PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT:
        mode_tag_suffix = f"{mode_tag_suffix}_nc{format_tag_token(noise_scope)}"
    if noise_granularity != PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT:
        mode_tag_suffix = f"{mode_tag_suffix}_ng{format_tag_token(noise_granularity)}"

    dataset_settings: Optional[PgasNoiseSettings] = None
    trial_settings: Optional[List[PgasNoiseSettings]] = None
    if noise_granularity == "trial":
        trial_settings = [
            _build_pgas_noise_settings(
                config=config,
                maxspikes=maxspikes,
                calibration_trials=[calibration_trial],
                spike_times=spike_times,
                input_fs=input_fs,
                bm_sigma_min=bm_sigma_min,
                bm_sigma_max=bm_sigma_max,
            )
            for calibration_trial in calibration_source_trials
        ]
        run_tag = f"{run_tag_base}_bmtrial{mode_tag_suffix}"
        bm_sigma_values = [float(setting.bm_sigma) for setting in trial_settings]
        sigma2_target_values = [
            float(setting.sigma2_target)
            for setting in trial_settings
            if setting.sigma2_target is not None
        ]
        sigma2_alpha_values = [
            float(setting.sigma2_alpha)
            for setting in trial_settings
            if setting.sigma2_alpha is not None
        ]
        sigma2_beta_values = [
            float(setting.sigma2_beta)
            for setting in trial_settings
            if setting.sigma2_beta is not None
        ]
        bm_sigma = float(np.median(bm_sigma_values))
        sigma2_target_effective = (
            float(np.median(sigma2_target_values)) if sigma2_target_values else None
        )
        sigma2_alpha_effective = (
            float(np.median(sigma2_alpha_values)) if sigma2_alpha_values else None
        )
        sigma2_beta_effective = (
            float(np.median(sigma2_beta_values)) if sigma2_beta_values else None
        )
        sigma2_prior_strength_effective = normalize_sigma2_prior_strength(
            config.sigma2_prior_strength
        )
        noise_calibration = None
    else:
        dataset_settings = _build_pgas_noise_settings(
            config=config,
            maxspikes=maxspikes,
            calibration_trials=calibration_source_trials,
            spike_times=spike_times,
            input_fs=input_fs,
            bm_sigma_min=bm_sigma_min,
            bm_sigma_max=bm_sigma_max,
        )
        bm_sigma = dataset_settings.bm_sigma
        sigma2_target_effective = dataset_settings.sigma2_target
        sigma2_alpha_effective = dataset_settings.sigma2_alpha
        sigma2_beta_effective = dataset_settings.sigma2_beta
        sigma2_prior_strength_effective = dataset_settings.sigma2_prior_strength
        noise_calibration = dataset_settings.noise_calibration
        bm_token = format_tag_token(f"{bm_sigma:.3g}")
        run_tag = f"{run_tag_base}_bm{bm_token}"
        if sigma2_target_effective is not None:
            s2_token = format_tag_token(f"{sigma2_target_effective:.3g}")
            run_tag = f"{run_tag}_s2{s2_token}"
            if config.sigma2_alpha is not None:
                a2_token = format_tag_token(f"{sigma2_alpha_effective:.3g}")
                run_tag = f"{run_tag}_a2{a2_token}"
            else:
                p2_token = format_tag_token(f"{sigma2_prior_strength_effective:.3g}")
                run_tag = f"{run_tag}_p2{p2_token}"
        elif sigma2_alpha_effective is not None:
            a2_token = format_tag_token(f"{sigma2_alpha_effective:.3g}")
            run_tag = f"{run_tag}_a2{a2_token}"
        run_tag = f"{run_tag}{mode_tag_suffix}"

    cfg_dict = {
        "niter": config.niter,
        "burnin": config.burnin,
        "downsample_target": config.downsample_label,
        "gparam_file": str(config.gparam_file),
        "maxspikes": maxspikes,
        "input_resample_fs": float(input_fs),
        "bm_sigma_bounds": {
            "min": float(bm_sigma_min),
            "max": float(bm_sigma_max),
        },
    }
    if dataset_settings is not None:
        cfg_dict.update(dataset_settings.cfg)
    else:
        cfg_dict["constants_file"] = "per-trial"
        cfg_dict["bm_sigma"] = bm_sigma
        cfg_dict["bm_sigma_by_trial"] = [
            float(setting.bm_sigma) for setting in trial_settings or []
        ]
        sigma2_by_trial = [
            (
                float(setting.sigma2_target)
                if setting.sigma2_target is not None
                else None
            )
            for setting in trial_settings or []
        ]
        cfg_dict["sigma2_target_by_trial"] = sigma2_by_trial
        cfg_dict["trial_noise_calibration"] = [
            {
                "trial_index": int(idx),
                **ensure_serializable(setting.cfg),
            }
            for idx, setting in enumerate(trial_settings or [])
        ]
        if sigma2_target_effective is not None:
            cfg_dict["sigma2_target"] = float(sigma2_target_effective)
        if sigma2_alpha_effective is not None:
            cfg_dict["sigma2_alpha"] = float(sigma2_alpha_effective)
        if sigma2_target_effective is not None:
            cfg_dict["sigma2_prior_strength"] = float(
                sigma2_prior_strength_effective
            )
    if (
        noise_scope != PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT
        or noise_granularity != PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT
    ):
        cfg_dict["noise_calibration_scope"] = noise_scope
        cfg_dict["noise_calibration_granularity"] = noise_granularity
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
            cached.metadata.setdefault(
                "bm_sigma_bounds",
                {"min": float(bm_sigma_min), "max": float(bm_sigma_max)},
            )
            if sigma2_target_effective is not None:
                cached.metadata.setdefault("sigma2_target", sigma2_target_effective)
            if sigma2_alpha_effective is not None:
                cached.metadata.setdefault("sigma2_alpha", sigma2_alpha_effective)
            if sigma2_beta_effective is not None:
                cached.metadata.setdefault("sigma2_beta", sigma2_beta_effective)
            if noise_calibration is not None:
                cached.metadata.setdefault("noise_calibration", noise_calibration)
            cached.metadata.setdefault("noise_calibration_scope", noise_scope)
            cached.metadata.setdefault("noise_calibration_granularity", noise_granularity)
            if trial_settings is not None:
                cached.metadata.setdefault(
                    "bm_sigma_by_trial",
                    [float(setting.bm_sigma) for setting in trial_settings],
                )
                cached.metadata.setdefault(
                    "sigma2_target_by_trial",
                    [
                        (
                            float(setting.sigma2_target)
                            if setting.sigma2_target is not None
                            else None
                        )
                        for setting in trial_settings
                    ],
                )
            return cached

        # Backwards-compatible cache lookup for older runs (e.g. `results/full_evaluation_by_run/base`)
        # whose PGAS caches were stored under `<dataset>_ms{maxspikes}` with numeric downsample targets.
        #
        # This keeps `--use-cache` effective even if the tagging/config signature evolved.
        if (
            config.resample_fs is None
            and noise_scope == PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT
            and noise_granularity == PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT
        ):
            legacy_tag = f"{config.dataset_tag}_ms{maxspikes}"
            if str(config.downsample_label).strip().lower() == "raw":
                legacy_downsample = "raw"
            else:
                match = re.search(r"(\d+(?:\.\d+)?)", str(config.downsample_label))
                legacy_downsample = (
                    f"{float(match.group(1)):.2f}" if match else str(config.downsample_label)
                )
            legacy_constants = prepare_constants_with_params(
                config.constants_file,
                maxspikes=maxspikes,
                bm_sigma=None,
            )
            legacy_cfg = {
                "niter": config.niter,
                "burnin": config.burnin,
                "downsample_target": legacy_downsample,
                "constants_file": str(legacy_constants),
                "gparam_file": str(config.gparam_file),
                "maxspikes": maxspikes,
            }
            if config.edges is not None:
                legacy_cfg["edge_hash"] = hash_array(config.edges)
            legacy_cached = load_method_cache("pgas", legacy_tag, legacy_cfg, trace_hash)
            if legacy_cached:
                legacy_cached.metadata.setdefault("maxspikes_per_bin", config.maxspikes_per_bin)
                legacy_cached.metadata.setdefault("cache_tag", legacy_tag)
                legacy_cached.metadata.setdefault("maxspikes", maxspikes)
                legacy_cached.metadata.setdefault("cache_style", "legacy")
                return legacy_cached

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
        if trial_settings is not None:
            constants_for_trial = trial_settings[trial_idx].constants_path
        elif dataset_settings is not None:
            constants_for_trial = dataset_settings.constants_path
        else:
            raise RuntimeError("PGAS noise settings were not initialized.")
        analyzer = pgas_mod.Analyzer(
            time=np.ascontiguousarray(trial.times, dtype=np.float64),
            data=np.ascontiguousarray(trial.values, dtype=np.float64),
            constants_file=str(constants_for_trial),
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
            seed=0, # should now pull from MCMC.seed in parameter_files
        )
        analyzer.run()

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
    traces.metadata.setdefault(
        "bm_sigma_bounds",
        {"min": float(bm_sigma_min), "max": float(bm_sigma_max)},
    )
    if sigma2_target_effective is not None:
        traces.metadata.setdefault("sigma2_target", sigma2_target_effective)
    if sigma2_alpha_effective is not None:
        traces.metadata.setdefault("sigma2_alpha", sigma2_alpha_effective)
    if sigma2_beta_effective is not None:
        traces.metadata.setdefault("sigma2_beta", sigma2_beta_effective)
    if noise_calibration is not None:
        traces.metadata.setdefault("noise_calibration", noise_calibration)
    traces.metadata.setdefault("noise_calibration_scope", noise_scope)
    traces.metadata.setdefault("noise_calibration_granularity", noise_granularity)
    if trial_settings is not None:
        traces.metadata.setdefault(
            "bm_sigma_by_trial",
            [float(setting.bm_sigma) for setting in trial_settings],
        )
        traces.metadata.setdefault(
            "sigma2_target_by_trial",
            [
                (
                    float(setting.sigma2_target)
                    if setting.sigma2_target is not None
                    else None
                )
                for setting in trial_settings
            ],
        )
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


def trim_trials_by_edges(
    trials: Sequence[TrialSeries],
    edges: np.ndarray,
    tolerance: float = 1e-6,
) -> List[TrialSeries]:
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


def load_pgas_component_series(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    burnin: int,
) -> Dict[str, np.ndarray]:
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


def load_pgas_method_result(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    burnin: int,
    metadata: Optional[Dict[str, object]] = None,
) -> MethodResult:
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
