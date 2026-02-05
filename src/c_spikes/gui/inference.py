from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import re

import numpy as np

from c_spikes.inference.cache import set_cache_root
from c_spikes.inference.cascade import CascadeConfig, run_cascade_inference, CASCADE_RESAMPLE_FS
from c_spikes.inference.ens2 import Ens2Config, run_ens2_inference
from c_spikes.inference.pgas import PgasConfig, run_pgas_inference
from c_spikes.inference.types import MethodResult, TrialSeries, compute_sampling_rate


@dataclass
class RunContext:
    data_dir: Path
    run_tag: str
    run_root: Path
    cache_root: Path
    pgas_output_root: Path
    pgas_temp_root: Path


@dataclass
class InferenceSettings:
    run_cascade: bool
    run_ens2: bool
    run_pgas: bool
    run_biophys: bool
    biophys_kind: str
    neuron_type: str
    use_cache: bool
    cascade_model_folder: Path
    cascade_model_name: str
    ens2_pretrained_dir: Path
    biophys_pretrained_dir: Path
    pgas_constants_file: Path
    pgas_gparam_file: Path
    cascade_discretize: bool = True
    cascade_resample_fs: float = CASCADE_RESAMPLE_FS
    pgas_resample_fs: Optional[float] = None
    pgas_fixed_bm_sigma: Optional[float] = None
    pgas_bm_sigma_gap_s: float = 0.15


def build_run_context(data_dir: Path, run_tag_input: str) -> RunContext:
    data_dir = Path(data_dir)
    run_tag = _normalize_run_tag(run_tag_input)
    if not run_tag:
        run_tag = _next_run_tag(data_dir)
    run_root = data_dir / f"c_spikes_run_{run_tag}"
    cache_root = run_root / "inference_cache"
    pgas_output_root = run_root / "pgas_output"
    pgas_temp_root = run_root / "pgas_temp"
    return RunContext(
        data_dir=data_dir,
        run_tag=run_tag,
        run_root=run_root,
        cache_root=cache_root,
        pgas_output_root=pgas_output_root,
        pgas_temp_root=pgas_temp_root,
    )


def ensure_run_dirs(context: RunContext) -> None:
    context.run_root.mkdir(parents=True, exist_ok=True)
    context.cache_root.mkdir(parents=True, exist_ok=True)
    context.pgas_output_root.mkdir(parents=True, exist_ok=True)
    context.pgas_temp_root.mkdir(parents=True, exist_ok=True)


def run_inference_for_epoch(
    *,
    epoch_id: str,
    time: np.ndarray,
    dff: np.ndarray,
    spike_times: Optional[np.ndarray],
    settings: InferenceSettings,
    context: RunContext,
    edges: Optional[np.ndarray] = None,
) -> Dict[str, MethodResult]:
    set_cache_root(context.cache_root)
    trial = _build_trial(time, dff)
    trials = [trial]
    raw_fs = compute_sampling_rate(trial.times)
    spikes = np.asarray(spike_times if spike_times is not None else np.array([]), dtype=np.float64).ravel()
    results: Dict[str, MethodResult] = {}

    if settings.run_pgas:
        pgas_cfg = PgasConfig(
            dataset_tag=epoch_id,
            output_root=context.pgas_output_root,
            constants_file=settings.pgas_constants_file,
            gparam_file=settings.pgas_gparam_file,
            resample_fs=settings.pgas_resample_fs,
            bm_sigma=settings.pgas_fixed_bm_sigma,
            bm_sigma_gap_s=settings.pgas_bm_sigma_gap_s,
            edges=edges,
            use_cache=settings.use_cache,
        )
        results["pgas"] = run_pgas_inference(
            trials=trials,
            raw_fs=raw_fs,
            spike_times=spikes,
            config=pgas_cfg,
        )

    if settings.run_ens2:
        ens2_cfg = Ens2Config(
            dataset_tag=epoch_id,
            pretrained_dir=settings.ens2_pretrained_dir,
            neuron_type=settings.neuron_type,
            downsample_label="raw",
            use_cache=settings.use_cache,
        )
        time_arr, trace_arr = _pad_trials(trials)
        results["ens2"] = run_ens2_inference(
            raw_time_stamps=time_arr,
            raw_traces=trace_arr,
            config=ens2_cfg,
            valid_lengths=[trial.times.size for trial in trials],
        )

    if settings.run_biophys:
        if settings.biophys_kind == "cascade":
            cascade_cfg = CascadeConfig(
                dataset_tag=epoch_id,
                model_folder=settings.biophys_pretrained_dir.parent,
                model_name=settings.biophys_pretrained_dir.name,
                resample_fs=settings.cascade_resample_fs,
                downsample_label="raw",
                use_cache=settings.use_cache,
                discretize=bool(settings.cascade_discretize),
            )
            biophys_result = run_cascade_inference(trials=trials, config=cascade_cfg)
            results["biophys_ml"] = _clone_method_result(
                biophys_result,
                "biophys_ml",
                extra_metadata={"base_method": "cascade"},
            )
        else:
            ens2_cfg = Ens2Config(
                dataset_tag=epoch_id,
                pretrained_dir=settings.biophys_pretrained_dir,
                neuron_type=settings.neuron_type,
                downsample_label="raw",
                use_cache=settings.use_cache,
            )
            time_arr, trace_arr = _pad_trials(trials)
            biophys_result = run_ens2_inference(
                raw_time_stamps=time_arr,
                raw_traces=trace_arr,
                config=ens2_cfg,
                valid_lengths=[trial.times.size for trial in trials],
            )
            results["biophys_ml"] = _clone_method_result(
                biophys_result,
                "biophys_ml",
                extra_metadata={"base_method": "ens2"},
            )

    if settings.run_cascade:
        cascade_cfg = CascadeConfig(
            dataset_tag=epoch_id,
            model_folder=settings.cascade_model_folder,
            model_name=settings.cascade_model_name,
            resample_fs=settings.cascade_resample_fs,
            downsample_label="raw",
            use_cache=settings.use_cache,
            discretize=bool(settings.cascade_discretize),
        )
        results["cascade"] = run_cascade_inference(trials=trials, config=cascade_cfg)

    return results


def run_inference_for_epoch_safe(
    *,
    epoch_id: str,
    time: np.ndarray,
    dff: np.ndarray,
    spike_times: Optional[np.ndarray],
    settings: InferenceSettings,
    context: RunContext,
    edges: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, MethodResult], Dict[str, str]]:
    set_cache_root(context.cache_root)
    trial = _build_trial(time, dff)
    trials = [trial]
    raw_fs = compute_sampling_rate(trial.times)
    spikes = np.asarray(spike_times if spike_times is not None else np.array([]), dtype=np.float64).ravel()
    results: Dict[str, MethodResult] = {}
    errors: Dict[str, str] = {}

    if settings.run_pgas:
        try:
            pgas_cfg = PgasConfig(
                dataset_tag=epoch_id,
                output_root=context.pgas_output_root,
                constants_file=settings.pgas_constants_file,
                gparam_file=settings.pgas_gparam_file,
                resample_fs=settings.pgas_resample_fs,
                bm_sigma=settings.pgas_fixed_bm_sigma,
                bm_sigma_gap_s=settings.pgas_bm_sigma_gap_s,
                edges=edges,
                use_cache=settings.use_cache,
            )
            results["pgas"] = run_pgas_inference(
                trials=trials,
                raw_fs=raw_fs,
                spike_times=spikes,
                config=pgas_cfg,
            )
        except Exception as exc:
            errors["pgas"] = str(exc)

    if settings.run_ens2:
        try:
            ens2_cfg = Ens2Config(
                dataset_tag=epoch_id,
                pretrained_dir=settings.ens2_pretrained_dir,
                neuron_type=settings.neuron_type,
                downsample_label="raw",
                use_cache=settings.use_cache,
            )
            time_arr, trace_arr = _pad_trials(trials)
            results["ens2"] = run_ens2_inference(
                raw_time_stamps=time_arr,
                raw_traces=trace_arr,
                config=ens2_cfg,
                valid_lengths=[trial.times.size for trial in trials],
            )
        except Exception as exc:
            errors["ens2"] = str(exc)

    if settings.run_biophys:
        try:
            if settings.biophys_kind == "cascade":
                cascade_cfg = CascadeConfig(
                    dataset_tag=epoch_id,
                    model_folder=settings.biophys_pretrained_dir.parent,
                    model_name=settings.biophys_pretrained_dir.name,
                    resample_fs=settings.cascade_resample_fs,
                    downsample_label="raw",
                    use_cache=settings.use_cache,
                    discretize=bool(settings.cascade_discretize),
                )
                biophys_result = run_cascade_inference(trials=trials, config=cascade_cfg)
                results["biophys_ml"] = _clone_method_result(
                    biophys_result,
                    "biophys_ml",
                    extra_metadata={"base_method": "cascade"},
                )
            else:
                ens2_cfg = Ens2Config(
                    dataset_tag=epoch_id,
                    pretrained_dir=settings.biophys_pretrained_dir,
                    neuron_type=settings.neuron_type,
                    downsample_label="raw",
                    use_cache=settings.use_cache,
                )
                time_arr, trace_arr = _pad_trials(trials)
                biophys_result = run_ens2_inference(
                    raw_time_stamps=time_arr,
                    raw_traces=trace_arr,
                    config=ens2_cfg,
                    valid_lengths=[trial.times.size for trial in trials],
                )
                results["biophys_ml"] = _clone_method_result(
                    biophys_result,
                    "biophys_ml",
                    extra_metadata={"base_method": "ens2"},
                )
        except Exception as exc:
            errors["biophys_ml"] = str(exc)

    if settings.run_cascade:
        try:
            cascade_cfg = CascadeConfig(
                dataset_tag=epoch_id,
                model_folder=settings.cascade_model_folder,
                model_name=settings.cascade_model_name,
                resample_fs=settings.cascade_resample_fs,
                downsample_label="raw",
                use_cache=settings.use_cache,
                discretize=bool(settings.cascade_discretize),
            )
            results["cascade"] = run_cascade_inference(trials=trials, config=cascade_cfg)
        except Exception as exc:
            errors["cascade"] = str(exc)

    return results, errors


def _normalize_run_tag(run_tag: str) -> str:
    token = str(run_tag).strip()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^A-Za-z0-9_\-]", "", token)
    return token


def _next_run_tag(data_dir: Path) -> str:
    data_dir = Path(data_dir)
    existing = []
    if data_dir.exists():
        for child in data_dir.iterdir():
            if not child.is_dir():
                continue
            match = re.match(r"c_spikes_run_(\d+)$", child.name)
            if match:
                existing.append(int(match.group(1)))
    if not existing:
        return "1"
    return str(max(existing) + 1)


def _build_trial(time: np.ndarray, dff: np.ndarray) -> TrialSeries:
    time = np.asarray(time, dtype=np.float64).ravel()
    dff = np.asarray(dff, dtype=np.float64).ravel()
    mask = np.isfinite(time) & np.isfinite(dff)
    time = time[mask]
    dff = dff[mask]
    if time.size < 2:
        raise ValueError("Not enough valid time points for inference.")
    return TrialSeries(times=time, values=dff)


def _pad_trials(trials: list[TrialSeries]) -> tuple[np.ndarray, np.ndarray]:
    lengths = [trial.times.size for trial in trials]
    max_len = max(lengths)
    time_arr = np.zeros((len(trials), max_len), dtype=np.float64)
    trace_arr = np.zeros_like(time_arr)
    for idx, trial in enumerate(trials):
        times = trial.times
        values = trial.values
        if times.size < max_len:
            dt = np.median(np.diff(times)) if times.size > 1 else 0.0
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
            pad = max_len - times.size
            extra_times = times[-1] + np.arange(1, pad + 1, dtype=np.float64) * dt
            extra_values = np.full(pad, values[-1], dtype=np.float64)
            times = np.concatenate([times, extra_times])
            values = np.concatenate([values, extra_values])
        time_arr[idx] = times
        trace_arr[idx] = values
    return time_arr, trace_arr


def _clone_method_result(
    result: MethodResult,
    name: str,
    *,
    extra_metadata: Optional[dict] = None,
) -> MethodResult:
    metadata = dict(result.metadata or {})
    if extra_metadata:
        metadata.update(extra_metadata)
    return MethodResult(
        name=name,
        time_stamps=np.asarray(result.time_stamps),
        spike_prob=np.asarray(result.spike_prob),
        sampling_rate=float(result.sampling_rate),
        metadata=metadata,
        reconstruction=None if result.reconstruction is None else np.asarray(result.reconstruction),
        discrete_spikes=None if result.discrete_spikes is None else np.asarray(result.discrete_spikes),
    )
