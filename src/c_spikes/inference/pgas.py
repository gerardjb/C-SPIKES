from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import re

from .cache import load_method_cache, save_method_cache
from .eval import segment_indices
from .pgas_cache import PgasSamplesCache, load_pgas_samples_from_cache
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
    bm_sigma_gap_s: float = 0.15
    edges: Optional[np.ndarray] = None
    use_cache: bool = True
    keep_output_dat_files: bool = False


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
    from .cache import get_cache_root
    from .types import ensure_serializable  # unused import hint

    cache_dir = get_cache_root() / "pgas_constants"
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
        trials_for_pgas = trim_trials_by_edges(trials, config.edges)
    else:
        trials_for_pgas = list(trials)

    resample_fs_val = config.resample_fs if config.resample_fs is not None else raw_fs

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
    bm_sigma = (
        config.bm_sigma
        if config.bm_sigma is not None
        else estimate_bm_sigma_for_trials(trials_for_pgas, spike_times, input_fs, config.bm_sigma_gap_s)
    )
    constants_path = prepare_constants_with_params(
        config.constants_file,
        maxspikes=maxspikes,
        bm_sigma=bm_sigma,
    )
    label_token = format_tag_token(config.downsample_label)
    pgas_resample_token = "raw" if config.resample_fs is None else format_tag_token(f"{config.resample_fs:g}")
    bm_token = format_tag_token(f"{bm_sigma:.3g}")
    run_tag = f"{config.dataset_tag}_s{label_token}_ms{maxspikes}_rs{pgas_resample_token}_bm{bm_token}"
    cfg_dict = {
        "niter": config.niter,
        "burnin": config.burnin,
        "downsample_target": config.downsample_label,
        "constants_file": str(constants_path),
        "gparam_file": str(config.gparam_file),
        "maxspikes": maxspikes,
        "input_resample_fs": float(input_fs),
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

        # Backwards-compatible cache lookup for older runs (e.g. `results/full_evaluation_by_run/base`)
        # whose PGAS caches were stored under `<dataset>_ms{maxspikes}` with numeric downsample targets.
        #
        # This keeps `--use-cache` effective even if the tagging/config signature evolved.
        if config.resample_fs is None:
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
    traces.metadata.setdefault("pgas_samples_cached", True)
    traces.metadata.setdefault("pgas_samples_schema_version", 1)
    traces.metadata.setdefault("pgas_output_dat_files_kept", bool(config.keep_output_dat_files))
    traces.metadata.setdefault(
        "pgas_output_dat_cleanup_policy",
        "keep" if config.keep_output_dat_files else "delete_after_cache_write",
    )
    traces.metadata.setdefault("cache_tag", run_tag)
    pgas_cache_payload = build_pgas_output_cache_payload(
        trials=trial_list,
        dataset_tag=run_tag,
        output_root=output_root,
        burnin=config.burnin,
    )
    save_method_cache(
        "pgas",
        run_tag,
        traces,
        cfg_dict,
        trace_hash,
        extra_payload=pgas_cache_payload,
    )
    if not config.keep_output_dat_files:
        cleanup_pgas_output_dat_files(
            output_root=output_root,
            dataset_tag=run_tag,
            n_trials=len(trial_list),
        )
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
    *,
    cache_mat_path: Optional[Path] = None,
    samples_cache: Optional[PgasSamplesCache] = None,
) -> Dict[str, np.ndarray]:
    if samples_cache is None and cache_mat_path is not None:
        try:
            samples_cache = load_pgas_samples_from_cache(cache_mat_path)
        except Exception:
            samples_cache = None

    spike_segments: List[np.ndarray] = []
    time_segments: List[np.ndarray] = []
    baseline_segments: List[np.ndarray] = []
    calcium_segments: List[np.ndarray] = []
    burst_segments: List[np.ndarray] = []
    map_segments: List[np.ndarray] = []
    for trial_idx, trial in enumerate(trials):
        cache_components = _load_pgas_trial_components_from_cache(
            samples_cache,
            trial_idx=trial_idx,
            burnin=burnin,
        )
        if cache_components is not None:
            trial_times = cache_components["time_stamps"]
            if trial_times.size == 0:
                trial_times = trial.times.copy()
            time_segments.append(np.asarray(trial_times, dtype=np.float64).ravel())
            spike_segments.append(cache_components["spikes_mean"])
            baseline_segments.append(cache_components["baseline_mean"])
            calcium_segments.append(cache_components["calcium_mean"])
            burst_segments.append(cache_components["burst_mean"])
            map_segments.append(cache_components["spikes_map"])
            continue

        tag = f"{dataset_tag}_trial{trial_idx}"
        dat_file = output_root / f"traj_samples_{tag}.dat"
        log_file = output_root / f"logp_{tag}.dat"
        if not dat_file.exists() or not log_file.exists():
            cache_suffix = (
                "" if cache_mat_path is None else f" or embedded PGAS samples in {cache_mat_path}"
            )
            raise FileNotFoundError(
                f"Missing PGAS output files for tag '{tag}'. Expected {dat_file} and {log_file}"
                f"{cache_suffix}."
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


def _load_pgas_trial_components_from_cache(
    samples_cache: Optional[PgasSamplesCache],
    *,
    trial_idx: int,
    burnin: int,
) -> Optional[Dict[str, np.ndarray]]:
    if samples_cache is None:
        return None
    traj = samples_cache.trajectory_for_trial(trial_idx)
    if traj is None and len(samples_cache.trajectory_samples) == 1 and trial_idx == 0:
        traj = samples_cache.trajectory_samples[0]
    if traj is None:
        return None

    burst_mat = _cache_matrix(traj.values, "burst")
    baseline_mat = _cache_matrix(traj.values, "baseline")
    spikes_mat = _cache_matrix(traj.values, "spikes")
    calcium_mat = _cache_matrix(traj.values, "calcium")
    n_samples = int(traj.n_samples or _first_nonempty_rows(burst_mat, baseline_mat, spikes_mat, calcium_mat))
    n_time = int(traj.n_time or _first_nonempty_cols(burst_mat, baseline_mat, spikes_mat, calcium_mat))
    if n_samples <= 0 or n_time <= 0:
        return None

    burnin_eff = int(max(0, min(int(burnin), n_samples - 1)))
    sample_slice = slice(burnin_eff, None)
    baseline_mean = baseline_mat[sample_slice, :].mean(axis=0)
    calcium_state_mean = calcium_mat[sample_slice, :].mean(axis=0)
    spikes_map = _cache_spikes_map(samples_cache, trial_idx, traj)
    if spikes_map is None:
        spikes_map = np.asarray(spikes_mat[sample_slice, :].mean(axis=0), dtype=np.float64)
    return {
        "time_stamps": np.asarray(traj.time_stamps, dtype=np.float64).ravel(),
        "spikes_mean": np.asarray(spikes_mat[sample_slice, :].mean(axis=0), dtype=np.float64),
        "baseline_mean": np.asarray(baseline_mean, dtype=np.float64),
        "burst_mean": np.asarray(burst_mat[sample_slice, :].mean(axis=0), dtype=np.float64),
        "calcium_mean": np.asarray(calcium_state_mean + baseline_mean, dtype=np.float64),
        "spikes_map": np.asarray(spikes_map, dtype=np.float64).ravel(),
    }


def _cache_spikes_map(
    samples_cache: PgasSamplesCache,
    trial_idx: int,
    traj: object,
) -> Optional[np.ndarray]:
    mapped = getattr(traj, "map", {}).get("spikes")
    if mapped is not None:
        arr = np.asarray(mapped, dtype=np.float64).ravel()
        if arr.size:
            return arr
    logp = samples_cache.logp_for_trial(trial_idx)
    map_idx = None if logp is None else logp.map_sample_index
    spikes_mat = _cache_matrix(getattr(traj, "values"), "spikes")
    if map_idx is None or map_idx < 0 or map_idx >= spikes_mat.shape[0]:
        return None
    return np.asarray(spikes_mat[int(map_idx), :], dtype=np.float64)


def _cache_matrix(values: Dict[str, np.ndarray], key: str) -> np.ndarray:
    arr = np.asarray(values.get(key, np.empty((0, 0))), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape((1, -1))
    return arr


def _first_nonempty_rows(*arrays: np.ndarray) -> int:
    for arr in arrays:
        if arr.ndim == 2 and arr.shape[0] > 0:
            return int(arr.shape[0])
    return 0


def _first_nonempty_cols(*arrays: np.ndarray) -> int:
    for arr in arrays:
        if arr.ndim == 2 and arr.shape[1] > 0:
            return int(arr.shape[1])
    return 0


def _as_row_object_array(items: Sequence[Dict[str, Any]]) -> np.ndarray:
    arr = np.empty((1, len(items)), dtype=object)
    for idx, item in enumerate(items):
        arr[0, idx] = item
    return arr


def _read_pgas_trajectory_file(dat_file: Path, burnin: int) -> Dict[str, Any]:
    with dat_file.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip()
    columns = [token.strip() for token in header.split(",") if token.strip()]
    data = np.genfromtxt(dat_file, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = np.asarray([data], dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0 or data.shape[1] < 5:
        raise ValueError(f"PGAS trajectory file has too few columns: {dat_file}")

    index = data[:, 0]
    time_bins = int(np.sum(index == 0))
    if time_bins <= 0:
        raise ValueError(f"PGAS trajectory file has no TIME axis: {dat_file}")
    n_samples = int(data.shape[0] // time_bins)
    if n_samples <= 0 or n_samples * time_bins != data.shape[0]:
        raise ValueError(
            f"PGAS trajectory rows {data.shape[0]} are not divisible by TIME={time_bins}: {dat_file}"
        )

    matrices: Dict[str, np.ndarray] = {}
    for col_idx, name in enumerate(columns):
        if col_idx >= data.shape[1]:
            continue
        matrices[name] = data[:, col_idx].reshape((n_samples, time_bins))
    if "Y" not in matrices:
        matrices["Y"] = np.full((n_samples, time_bins), np.nan, dtype=np.float64)

    burnin_eff = int(max(0, min(int(burnin), n_samples - 1)))
    sample_slice = slice(burnin_eff, None)
    post_mean = {
        name: values[sample_slice, :].mean(axis=0)
        for name, values in matrices.items()
        if name != "index"
    }
    baseline_matrix = matrices.get("B", np.full((n_samples, time_bins), np.nan))
    calcium_matrix = matrices.get("C", np.full((n_samples, time_bins), np.nan))
    reconstruction_mean = calcium_matrix[sample_slice, :].mean(axis=0) + baseline_matrix[
        sample_slice, :
    ].mean(axis=0)

    return {
        "path": str(dat_file),
        "columns": np.asarray(columns, dtype=object),
        "n_samples": np.asarray([[n_samples]], dtype=np.int64),
        "n_time": np.asarray([[time_bins]], dtype=np.int64),
        "burnin": np.asarray([[burnin_eff]], dtype=np.int64),
        "index": matrices.get("index", np.empty((n_samples, time_bins))),
        "burst": matrices.get("burst", np.full((n_samples, time_bins), np.nan)),
        "baseline": matrices.get("B", np.full((n_samples, time_bins), np.nan)),
        "spikes": matrices.get("S", np.full((n_samples, time_bins), np.nan)),
        "calcium": matrices.get("C", np.full((n_samples, time_bins), np.nan)),
        "observation": matrices.get("Y", np.full((n_samples, time_bins), np.nan)),
        "post_burnin_mean": {
            "burst": post_mean.get("burst", np.full(time_bins, np.nan)),
            "baseline": post_mean.get("B", np.full(time_bins, np.nan)),
            "spikes": post_mean.get("S", np.full(time_bins, np.nan)),
            "calcium": post_mean.get("C", np.full(time_bins, np.nan)),
            "reconstruction": reconstruction_mean,
            "observation": post_mean.get("Y", np.full(time_bins, np.nan)),
        },
    }


def _read_pgas_param_file(param_file: Path, burnin: int) -> Dict[str, Any]:
    with param_file.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip()
    columns = [token.strip() for token in header.split(",") if token.strip()]
    data = np.genfromtxt(param_file, delimiter=",", skip_header=1)
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if data.size == 0:
        data = np.empty((0, len(columns)), dtype=np.float64)
    n_samples = int(data.shape[0])
    burnin_eff = int(max(0, min(int(burnin), n_samples - 1))) if n_samples else 0
    post = data[burnin_eff:, :] if n_samples else data
    return {
        "path": str(param_file),
        "columns": np.asarray(columns, dtype=object),
        "values": data,
        "burnin": np.asarray([[burnin_eff]], dtype=np.int64),
        "post_burnin_mean": post.mean(axis=0) if post.size else np.full(len(columns), np.nan),
    }


def _read_pgas_logp_file(log_file: Path, burnin: int) -> Dict[str, Any]:
    values = np.atleast_1d(np.genfromtxt(log_file))
    values = np.asarray(values, dtype=np.float64).ravel()
    n_samples = int(values.size)
    burnin_eff = int(max(0, min(int(burnin), n_samples - 1))) if n_samples else 0
    post = values[burnin_eff:] if n_samples else values
    if post.size:
        map_offset = int(np.argmax(post))
        map_index = int(map_offset + burnin_eff)
        map_logp = float(values[map_index])
    else:
        map_index = -1
        map_logp = np.nan
    return {
        "path": str(log_file),
        "values": values,
        "burnin": np.asarray([[burnin_eff]], dtype=np.int64),
        "post_burnin_mean": np.asarray([[float(post.mean()) if post.size else np.nan]]),
        "post_burnin_max": np.asarray([[map_logp]]),
        "map_sample_index": np.asarray([[map_index]], dtype=np.int64),
    }


def build_pgas_output_cache_payload(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    burnin: int,
) -> Dict[str, Any]:
    trajectory_items: List[Dict[str, Any]] = []
    parameter_items: List[Dict[str, Any]] = []
    logp_items: List[Dict[str, Any]] = []
    for trial_idx, trial in enumerate(trials):
        tag = f"{dataset_tag}_trial{trial_idx}"
        traj_file = Path(output_root) / f"traj_samples_{tag}.dat"
        param_file = Path(output_root) / f"param_samples_{tag}.dat"
        log_file = Path(output_root) / f"logp_{tag}.dat"
        if not traj_file.exists() or not param_file.exists() or not log_file.exists():
            raise FileNotFoundError(
                "Missing PGAS output files for cache payload. Expected "
                f"{traj_file}, {param_file}, and {log_file}."
            )
        trajectory = _read_pgas_trajectory_file(traj_file, burnin)
        trajectory["tag"] = tag
        trajectory["trial_index"] = np.asarray([[trial_idx]], dtype=np.int64)
        trajectory["time_stamps"] = np.asarray(trial.times, dtype=np.float64)
        parameters = _read_pgas_param_file(param_file, burnin)
        parameters["tag"] = tag
        parameters["trial_index"] = np.asarray([[trial_idx]], dtype=np.int64)
        logp = _read_pgas_logp_file(log_file, burnin)
        logp["tag"] = tag
        logp["trial_index"] = np.asarray([[trial_idx]], dtype=np.int64)

        map_idx = int(np.asarray(logp["map_sample_index"]).squeeze())
        n_samples = int(np.asarray(trajectory["n_samples"]).squeeze())
        if 0 <= map_idx < n_samples:
            trajectory["map"] = {
                "sample_index": np.asarray([[map_idx]], dtype=np.int64),
                "logp": np.asarray([[float(np.asarray(logp["post_burnin_max"]).squeeze())]]),
                "burst": trajectory["burst"][map_idx, :],
                "baseline": trajectory["baseline"][map_idx, :],
                "spikes": trajectory["spikes"][map_idx, :],
                "calcium": trajectory["calcium"][map_idx, :],
                "reconstruction": trajectory["calcium"][map_idx, :] + trajectory["baseline"][map_idx, :],
                "observation": trajectory["observation"][map_idx, :],
            }

        trajectory_items.append(trajectory)
        parameter_items.append(parameters)
        logp_items.append(logp)

    return {
        "pgas_samples": {
            "schema_version": np.asarray([[1]], dtype=np.int64),
            "description": "Full PGAS .dat outputs packed into MATLAB structs; trajectory matrices are n_samples x n_time.",
            "trajectory_samples": _as_row_object_array(trajectory_items),
            "parameter_samples": _as_row_object_array(parameter_items),
            "logp": _as_row_object_array(logp_items),
        }
    }


def cleanup_pgas_output_dat_files(
    *,
    output_root: Path,
    dataset_tag: str,
    n_trials: int,
) -> List[Path]:
    """Remove raw PGAS sample dumps after the cache .mat has been written.

    ``last_params_*.dat`` files are intentionally preserved because they are
    small and useful for partial-progress recovery/debugging if a run fails.
    """

    removed: List[Path] = []
    for trial_idx in range(int(n_trials)):
        tag = f"{dataset_tag}_trial{trial_idx}"
        for prefix in ("traj_samples", "param_samples", "logp"):
            path = Path(output_root) / f"{prefix}_{tag}.dat"
            if not path.exists():
                continue
            try:
                path.unlink()
                removed.append(path)
            except OSError as exc:
                print(f"[PGAS cache] Warning: could not remove raw output {path}: {exc}")
    return removed


def load_pgas_method_result(
    trials: Sequence[TrialSeries],
    dataset_tag: str,
    output_root: Path,
    burnin: int,
    metadata: Optional[Dict[str, object]] = None,
    *,
    cache_mat_path: Optional[Path] = None,
    samples_cache: Optional[PgasSamplesCache] = None,
) -> MethodResult:
    traces = load_pgas_component_series(
        trials,
        dataset_tag,
        output_root,
        burnin,
        cache_mat_path=cache_mat_path,
        samples_cache=samples_cache,
    )
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
