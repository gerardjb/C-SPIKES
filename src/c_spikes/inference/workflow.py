from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from c_spikes.utils import load_Janelia_data

from .cascade import CASCADE_RESAMPLE_FS, CascadeConfig, run_cascade_inference
from .ens2 import Ens2Config, run_ens2_inference
from .eval import build_ground_truth_series, compute_correlations
from .pgas import (
    PGAS_RESAMPLE_FS,
    PGAS_BURNIN,
    PGAS_NITER,
    PGAS_BM_SIGMA_DEFAULT,
    PgasConfig,
    run_pgas_inference,
    pgas_windows_from_result,
    trim_trials_by_edges,
)
from .smoothing import mean_downsample_trace, resample_trials_to_fs
from .types import MethodResult, TrialSeries, ensure_serializable, extract_spike_times, flatten_trials


@dataclass
class SmoothingLevel:
    target_fs: Optional[float]
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label is None:
            if self.target_fs is None:
                self.label = "raw"
            else:
                self.label = f"{self.target_fs:.1f}Hz"


@dataclass
class MethodSelection:
    run_pgas: bool = True
    run_ens2: bool = True
    run_cascade: bool = True


@dataclass
class DatasetRunConfig:
    dataset_path: Path
    neuron_type: str = "Exc"
    smoothing: SmoothingLevel = field(default_factory=lambda: SmoothingLevel(target_fs=None))
    reference_fs: Optional[float] = None
    edges: Optional[np.ndarray] = None
    selection: MethodSelection = field(default_factory=MethodSelection)
    use_cache: bool = True
    bm_sigma_gap_s: float = 0.15
    pgas_resample_fs: Optional[float] = None  # None => use raw/native
    cascade_resample_fs: Optional[float] = None  # None => default CASCADE_RESAMPLE_FS
    pgas_fixed_bm_sigma: Optional[float] = None  # Optional fixed bm_sigma (skip tuning)


def run_inference_for_dataset(
    cfg: DatasetRunConfig,
    *,
    pgas_constants: Path,
    pgas_gparam: Path,
    pgas_output_root: Path,
    ens2_pretrained_root: Path,
    cascade_model_root: Path,
    dataset_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict[str, object]:
    dataset_tag = cfg.dataset_path.stem
    if dataset_data is None:
        time_stamps, dff, spike_times = load_Janelia_data(str(cfg.dataset_path))
    else:
        time_stamps, dff, spike_times = dataset_data
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()

    trials_native: List[TrialSeries] = []
    for idx in range(time_stamps.shape[0]):
        t = np.asarray(time_stamps[idx], dtype=np.float64)
        y = np.asarray(dff[idx], dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        if t.size == 0:
            continue
        trials_native.append(TrialSeries(times=t, values=y))
    if not trials_native:
        raise RuntimeError(f"No valid trials for dataset {dataset_tag}.")

    if cfg.edges is not None:
        if cfg.edges.shape[0] != len(trials_native):
            raise ValueError(
                f"Edges shape {cfg.edges.shape} does not match {len(trials_native)} trials after slicing."
            )
        trials_native = trim_trials_by_edges(trials_native, cfg.edges)

    raw_time_flat, raw_trace_flat = flatten_trials(trials_native)
    raw_fs = 1.0 / np.median(np.diff(raw_time_flat))

    if cfg.smoothing.target_fs is None:
        trials_for_methods = trials_native
        down_time_flat, down_trace_flat = raw_time_flat, raw_trace_flat
        downsample_label = cfg.smoothing.label or "raw"
    else:
        trials_for_methods = [
            mean_downsample_trace(trial.times, trial.values, cfg.smoothing.target_fs)
            for trial in trials_native
        ]
        down_time_flat, down_trace_flat = flatten_trials(trials_for_methods)
        downsample_label = cfg.smoothing.label or f"{cfg.smoothing.target_fs:.2f}"

    methods: Dict[str, MethodResult] = {}

    if cfg.selection.run_pgas:
        pgas_cfg = PgasConfig(
            dataset_tag=dataset_tag,
            output_root=pgas_output_root,
            constants_file=pgas_constants,
            gparam_file=pgas_gparam,
            resample_fs=cfg.pgas_resample_fs,
            niter=PGAS_NITER,
            burnin=PGAS_BURNIN,
            downsample_label=downsample_label,
            maxspikes=None,
            bm_sigma=cfg.pgas_fixed_bm_sigma if cfg.pgas_fixed_bm_sigma is not None else PGAS_BM_SIGMA_DEFAULT,
            bm_sigma_gap_s=cfg.bm_sigma_gap_s,
            edges=cfg.edges,
            use_cache=cfg.use_cache,
        )
        pgas_result = run_pgas_inference(
            trials_for_methods,
            raw_fs=raw_fs,
            spike_times=spike_times,
            config=pgas_cfg,
        )
        methods["pgas"] = pgas_result
    else:
        pgas_result = None

    if cfg.selection.run_ens2:
        ens2_cfg = Ens2Config(
            dataset_tag=dataset_tag,
            pretrained_dir=ens2_pretrained_root,
            neuron_type=cfg.neuron_type,
            downsample_label=downsample_label,
            use_cache=cfg.use_cache,
        )
        ens2_result = run_ens2_inference(
            raw_time_stamps=time_stamps,
            raw_traces=dff,
            config=ens2_cfg,
        )
        methods["ens2"] = ens2_result
    else:
        ens2_result = None

    if cfg.selection.run_cascade:
        cascade_fs = cfg.cascade_resample_fs or CASCADE_RESAMPLE_FS
        cascade_trials = resample_trials_to_fs(trials_for_methods, cascade_fs)
        cascade_cfg = CascadeConfig(
            dataset_tag=dataset_tag,
            model_folder=cascade_model_root,
            model_name="Cascade_Universal_30Hz",
            resample_fs=cascade_fs,
            downsample_label=downsample_label,
            use_cache=cfg.use_cache,
        )
        cascade_result = run_cascade_inference(
            trials=cascade_trials,
            config=cascade_cfg,
        )
        methods["cascade"] = cascade_result
    else:
        cascade_result = None

    if not methods:
        return {
            "dataset": dataset_tag,
            "methods": {},
            "correlations": {},
            "spike_times": {},
            "summary": {},
        }

    if cfg.reference_fs is not None:
        ref_fs = cfg.reference_fs
    elif cfg.smoothing.target_fs is not None:
        ref_fs = cfg.smoothing.target_fs
    else:
        ref_fs = raw_fs

    global_start = min(trial.times[0] for trial in trials_native)
    global_end = max(trial.times[-1] for trial in trials_native)
    ref_time, ref_trace = build_ground_truth_series(
        spike_times, global_start, global_end, reference_fs=ref_fs
    )

    windows: Optional[List[Tuple[float, float]]] = None
    if pgas_result is not None:
        windows = pgas_windows_from_result(pgas_result)

    method_list = list(methods.values())
    correlations = compute_correlations(
        method_list,
        ref_time,
        ref_trace,
        windows=windows,
    )

    spike_times_dict: Dict[str, np.ndarray] = {}
    for label, result in methods.items():
        spikes = extract_spike_times(result)
        spike_times_dict[label] = spikes if spikes is not None else np.array([])

    summary = {
        "dataset": dataset_tag,
        "smoothing": cfg.smoothing.label,
        "downsample_target": downsample_label,
        "correlations": {k: float(v) for k, v in correlations.items()},
        "pgas_input_resample_fs": PGAS_RESAMPLE_FS if pgas_result is not None else None,
        "cascade_input_resample_fs": CASCADE_RESAMPLE_FS if cascade_result is not None else None,
        "gt_count": int(spike_times.size),
    }
    if pgas_result is not None:
        summary.update(
            {
                "pgas_cache": ensure_serializable(pgas_result.metadata.get("config", {}))
                if pgas_result.metadata
                else {},
                "pgas_maxspikes": pgas_result.metadata.get("maxspikes"),
                "pgas_maxspikes_per_bin": pgas_result.metadata.get("maxspikes_per_bin"),
            }
        )
    if ens2_result is not None:
        summary["ens2_cache"] = ensure_serializable(ens2_result.metadata.get("config", {}))
    if cascade_result is not None:
        summary["cascade_cache"] = ensure_serializable(cascade_result.metadata.get("config", {}))

    return {
        "dataset": dataset_tag,
        "methods": methods,
        "correlations": correlations,
        "spike_times": spike_times_dict,
        "summary": summary,
        "raw_time": raw_time_flat,
        "raw_trace": raw_trace_flat,
        "down_time": down_time_flat,
        "down_trace": down_trace_flat,
        "reference_time": ref_time,
        "reference_trace": ref_trace,
        "windows": windows,
    }
