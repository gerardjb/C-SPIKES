"""
Reusable batching/orchestration layer for running spike-inference methods.

This wraps the existing compare_inference_methods helpers so callers can:
  • select any subset of methods (pgas / ens2 / cascade / oasis),
  • batch over datasets via globbing or explicit lists,
  • drive runs either from Python (direct call) or the CLI wrapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from c_spikes.inference.cascade import CASCADE_RESAMPLE_FS
from c_spikes.inference.cache import get_cache_root
from c_spikes.inference.pgas import (
    PGAS_BM_SIGMA_DEFAULT,
    PGAS_BM_SIGMA_MAX,
    PGAS_BM_SIGMA_MIN,
    PGAS_NOISE_CALIBRATION_METHOD_DEFAULT,
    PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT,
    PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT,
    PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT,
)
from c_spikes.inference.smoothing import resolve_smoothing_levels
from c_spikes.inference.types import MethodResult, compute_config_signature, ensure_serializable
from c_spikes.inference.workflow import (
    DatasetRunConfig,
    MethodSelection,
    SmoothingLevel,
    run_inference_for_dataset,
)


DEFAULT_EDGES_PATH = Path("results/excitatory_time_stamp_edges.npy")


@dataclass
class RunConfig:
    data_root: Path = Path("data/janelia_8f/excitatory")
    dataset_glob: str = "*.mat"
    datasets: Optional[List[str]] = None  # stems without .mat
    max_datasets: Optional[int] = None
    smoothing_levels: Optional[Sequence[str]] = None  # tokens understood by resolve_smoothing_levels
    output_root: Path = Path("results/full_evaluation")
    edges_path: Path = DEFAULT_EDGES_PATH
    methods: Sequence[str] = ("pgas", "ens2", "cascade")
    neuron_type: str = "Exc"
    use_cache: bool = False
    first_trial_only: bool = False
    trial_selection_path: Optional[Path] = None
    bm_sigma_spike_gap: float = 0.15
    corr_sigma_ms: float = 50.0
    pgas_constants: Path = Path("parameter_files/constants_GCaMP8_soma.json")
    pgas_gparam: Path = Path("src/c_spikes/pgas/20230525_gold.dat")
    pgas_output_root: Path = Path("results/pgas_output/comparison")
    pgas_resample_fs: Optional[float] = None
    cascade_resample_fs: Optional[float] = None  # None => use input sampling rate (no forced resample)
    cascade_discretize: bool = True
    ens2_pretrained_root: Path = Path("results/Pretrained_models/ens2_published")
    cascade_model_root: Path = Path("results/Pretrained_models")
    cascade_model_name: str = "universal_p_cascade_exc_30"
    pgas_maxspikes: Optional[int] = None
    pgas_fixed_bm_sigma: Optional[float] = PGAS_BM_SIGMA_DEFAULT
    pgas_bm_sigma_min: float = PGAS_BM_SIGMA_MIN
    pgas_bm_sigma_max: float = PGAS_BM_SIGMA_MAX
    pgas_keep_output_dat_files: bool = False
    pgas_bm_sigma_use_low_activity_mask: bool = False
    pgas_sigma2_target: Optional[float] = None
    pgas_sigma2_alpha: Optional[float] = None
    pgas_sigma2_prior_strength: float = PGAS_SIGMA2_PRIOR_STRENGTH_DEFAULT
    pgas_noise_calibration_scope: str = PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT
    pgas_noise_calibration_granularity: str = PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT
    pgas_noise_calibration_method: str = PGAS_NOISE_CALIBRATION_METHOD_DEFAULT
    run_tag: Optional[str] = None  # optional override
    pgas_c0_first_y: bool = False
    trialwise_correlations: bool = False
    eval_only: bool = False
    oasis_g: Tuple[Optional[float], ...] = (None,)
    oasis_sn: Optional[float] = None
    oasis_b: Optional[float] = None
    oasis_b_nonneg: bool = True
    oasis_optimize_g: int = 0
    oasis_penalty: int = 1
    oasis_decimate: int = 1
    oasis_max_iter: Optional[int] = None
    oasis_shift: Optional[int] = None
    oasis_window: Optional[int] = None
    oasis_tol: Optional[float] = None
    oasis_uniformity_rtol: float = 5e-3
    oasis_uniformity_atol: float = 1e-9


def _select_dataset_paths(cfg: RunConfig) -> List[Path]:
    if cfg.datasets:
        paths = [
            cfg.data_root / (stem if str(stem).endswith(".mat") else f"{stem}.mat")
            for stem in cfg.datasets
        ]
    else:
        paths = sorted(cfg.data_root.glob(cfg.dataset_glob))
    if cfg.max_datasets is not None:
        paths = paths[: cfg.max_datasets]
    return paths


def _build_run_tag(cfg: RunConfig) -> str:
    if cfg.run_tag:
        return cfg.run_tag
    tokens: List[str] = []
    methods = {m.lower() for m in cfg.methods}
    if "pgas" in methods:
        if cfg.pgas_resample_fs is None:
            pgas_token = "pgasraw"
        else:
            pgas_token = f"pgas{_format_token(cfg.pgas_resample_fs)}"
        if cfg.pgas_maxspikes is not None:
            pgas_token = f"{pgas_token}_ms{cfg.pgas_maxspikes}"
        if cfg.pgas_c0_first_y:
            pgas_token = f"{pgas_token}_c0y"
        tokens.append(pgas_token)
    if "cascade" in methods:
        if cfg.cascade_resample_fs is None:
            cascade_token = "cascadein"
        else:
            cascade_token = f"cascade{_format_token(cfg.cascade_resample_fs)}"
        if not cfg.cascade_discretize:
            cascade_token = f"{cascade_token}_nodisc"
        tokens.append(cascade_token)
    if "ens2" in methods:
        tokens.append("ens2")
    if "oasis" in methods:
        oasis_g = tuple(cfg.oasis_g)
        if len(oasis_g) not in (1, 2):
            raise ValueError("oasis_g must contain one AR(1) or two AR(2) coefficients")
        oasis_settings = {
            "g": oasis_g,
            "sn": cfg.oasis_sn,
            "b": cfg.oasis_b,
            "b_nonneg": cfg.oasis_b_nonneg,
            "optimize_g": cfg.oasis_optimize_g,
            "penalty": cfg.oasis_penalty,
            "decimate": cfg.oasis_decimate,
            "max_iter": cfg.oasis_max_iter,
            "shift": cfg.oasis_shift,
            "window": cfg.oasis_window,
            "tol": cfg.oasis_tol,
            "uniformity_rtol": cfg.oasis_uniformity_rtol,
            "uniformity_atol": cfg.oasis_uniformity_atol,
        }
        oasis_signature, _ = compute_config_signature(oasis_settings)
        tokens.append(f"oasisar{len(oasis_g)}_{oasis_signature[:8]}")
    return "_".join(tokens) if tokens else "no_methods"


def _format_token(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return str(value).replace(".", "p")


def _normalize_methods(methods: Iterable[str]) -> List[str]:
    normalized = []
    for name in methods:
        token = name.strip().lower()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _count_samples(discrete_spikes: object) -> int:
    """
    Convert a per-sample spike series into an integer "sample count" for summaries.

    Some backends may yield arrays containing NaNs (e.g. padding/misalignment artifacts);
    treat those as missing and avoid crashing the batch run.
    """
    if discrete_spikes is None:
        return 0
    arr = np.asarray(discrete_spikes)
    if arr.size == 0:
        return 0
    total = float(np.nansum(arr.astype(np.float64, copy=False)))
    if not np.isfinite(total):
        return 0
    return int(total)


def _load_trial_selection(path: Path) -> Dict[str, List[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Trial selection JSON must be an object: {dataset_stem: [trial_idx, ...]}.")
    out: Dict[str, List[int]] = {}
    for key, value in payload.items():
        dataset = str(key).strip()
        if not dataset:
            continue
        if not isinstance(value, list):
            raise ValueError(
                f"Trial selection for '{dataset}' must be a list of non-negative integers."
            )
        indices: List[int] = []
        seen: set[int] = set()
        for item in value:
            try:
                idx = int(item)
            except Exception as exc:
                raise ValueError(
                    f"Invalid trial index '{item}' for dataset '{dataset}'."
                ) from exc
            if idx < 0:
                raise ValueError(
                    f"Trial index must be non-negative for dataset '{dataset}': {idx}."
                )
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
        out[dataset] = sorted(indices)
    return out


def run_batch(cfg: RunConfig) -> List[Path]:
    """
    Run the selected methods across datasets/smoothing levels and emit summaries.

    Returns:
        List[Path]: Paths to the summary.json files written.
    """
    run_tag = _build_run_tag(cfg)
    method_list = _normalize_methods(cfg.methods)
    smoothing_levels = resolve_smoothing_levels(cfg.smoothing_levels)
    dataset_paths = _select_dataset_paths(cfg)
    if not dataset_paths:
        raise FileNotFoundError(f"No datasets matched under {cfg.data_root} with pattern {cfg.dataset_glob}")

    edges_lookup = None
    if cfg.edges_path.exists():
        edges_lookup = np.load(cfg.edges_path, allow_pickle=True).item()

    trial_selection_lookup: Optional[Dict[str, List[int]]] = None
    if cfg.trial_selection_path is not None:
        if not cfg.trial_selection_path.exists():
            raise FileNotFoundError(cfg.trial_selection_path)
        trial_selection_lookup = _load_trial_selection(cfg.trial_selection_path)

    summaries: List[Path] = []
    for dataset_path in dataset_paths:
        dataset_tag = dataset_path.stem
        selected_trial_indices: Optional[List[int]] = None
        selected_trial_set: Optional[set[int]] = None
        if trial_selection_lookup is not None:
            selected_trial_indices = trial_selection_lookup.get(dataset_tag)
            if selected_trial_indices is None:
                continue
        if cfg.first_trial_only:
            selected_trial_indices = (
                selected_trial_indices[:1]
                if selected_trial_indices is not None
                else [0]
            )
        if selected_trial_indices is not None:
            selected_trial_set = set(selected_trial_indices)
        for label, target in smoothing_levels:
            selection = MethodSelection(
                run_pgas=("pgas" in method_list),
                run_ens2=("ens2" in method_list),
                run_cascade=("cascade" in method_list),
                run_oasis=("oasis" in method_list),
            )
            edges = None
            if edges_lookup is not None and dataset_tag in edges_lookup:
                edges = np.asarray(edges_lookup[dataset_tag], dtype=np.float64)
            smoothing = SmoothingLevel(label=label, target_fs=target)

            summary_dir = cfg.output_root / run_tag / dataset_tag / label
            summary_path = summary_dir / "summary.json"
            manifest_path = summary_dir / "comparison.json"
            if cfg.eval_only:
                if not manifest_path.exists() or not summary_path.exists():
                    print(f"[eval-only] Missing {summary_path} or {manifest_path}; skipping.")
                    continue
                from c_spikes.inference.eval import (
                    compute_epochwise_counts,
                    compute_correlations_windowed,
                    compute_trialwise_correlations_windowed,
                )
                from c_spikes.inference.types import MethodResult, TrialSeries, compute_config_signature, compute_sampling_rate
                from c_spikes.utils import load_Janelia_data

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                entries = manifest.get("methods", [])
                if not isinstance(entries, list) or not entries:
                    print(f"[eval-only] No method entries in {manifest_path}; skipping.")
                    continue

                # Load dataset spikes and compute a raw sampling rate (needed for 'raw' reference_fs).
                time_stamps, dff, spike_times = load_Janelia_data(str(dataset_path))
                spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
                trials: List[TrialSeries] = []
                for i in range(time_stamps.shape[0]):
                    if selected_trial_set is not None and i not in selected_trial_set:
                        continue
                    t = np.asarray(time_stamps[i], dtype=np.float64).ravel()
                    y = np.asarray(dff[i], dtype=np.float64).ravel()
                    m = np.isfinite(t) & np.isfinite(y)
                    t = t[m]
                    y = y[m]
                    if edges is not None and i < edges.shape[0]:
                        s, e = edges[i]
                        if np.isfinite(s) and np.isfinite(e) and e > s:
                            mwin = (t >= float(s)) & (t <= float(e))
                            t = t[mwin]
                            y = y[mwin]
                    if t.size:
                        trials.append(TrialSeries(times=t, values=y))
                raw_time = np.concatenate([tr.times for tr in trials]) if trials else np.asarray([], dtype=np.float64)
                raw_fs = float(compute_sampling_rate(raw_time)) if raw_time.size else float("nan")
                reference_fs = float(target) if target is not None else raw_fs
                if not np.isfinite(reference_fs) or reference_fs <= 0:
                    print(f"[eval-only] Invalid reference_fs for {dataset_tag}/{label}; skipping.")
                    continue

                # Determine evaluation windows.
                if edges is not None and selected_trial_indices is not None:
                    windows = [
                        (float(edges[idx, 0]), float(edges[idx, 1]))
                        for idx in selected_trial_indices
                        if 0 <= int(idx) < edges.shape[0]
                    ]
                elif edges is not None:
                    windows = [(float(s), float(e)) for s, e in edges]
                else:
                    windows = None
                if windows is None:
                    # Fall back to windows recorded in summary, else use full trials.
                    existing = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and isinstance(existing.get("trial_windows_s"), list):
                        windows = [(float(s), float(e)) for s, e in existing["trial_windows_s"]]  # type: ignore[index]
                    else:
                        windows = [(float(tr.times[0]), float(tr.times[-1])) for tr in trials]

                # Load cached method outputs from the configured inference cache root.
                cache_root = get_cache_root()
                loaded: Dict[str, MethodResult] = {}
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    method_name = str(entry.get("method", "")).strip().lower()
                    if method_name not in method_list:
                        continue
                    cache_tag = entry.get("cache_tag")
                    cache_tag = "" if cache_tag is None else str(cache_tag).strip()
                    if cache_tag.lower() in {"none", ""}:
                        cache_tag = dataset_tag
                    cache_key = entry.get("cache_key")
                    cache_key = "" if cache_key is None else str(cache_key).strip()
                    if not cache_key:
                        cfg_dict = entry.get("config", {})
                        if isinstance(cfg_dict, dict):
                            cache_key, _ = compute_config_signature(cfg_dict)
                    if not cache_key:
                        continue
                    mat_path = cache_root / method_name / cache_tag / f"{cache_key}.mat"
                    if not mat_path.exists():
                        print(f"[eval-only] Missing cache mat: {mat_path}")
                        continue
                    import scipy.io as sio

                    data = sio.loadmat(mat_path)
                    time_arr = np.asarray(data.get("time_stamps")).squeeze()
                    prob_arr = np.asarray(data.get("spike_prob")).squeeze()
                    reconstruction_arr = data.get("reconstruction")
                    reconstruction_arr = (
                        None
                        if reconstruction_arr is None
                        else np.asarray(reconstruction_arr).squeeze()
                    )
                    discrete_arr = data.get("discrete_spikes")
                    discrete_arr = (
                        None if discrete_arr is None else np.asarray(discrete_arr).squeeze()
                    )
                    sampling_rate = float(entry.get("sampling_rate", 0.0) or 0.0)
                    if sampling_rate <= 0:
                        sampling_rate = float(compute_sampling_rate(np.asarray(time_arr, dtype=np.float64).ravel()))
                    loaded[method_name] = MethodResult(
                        name=method_name,
                        time_stamps=time_arr,
                        spike_prob=prob_arr,
                        sampling_rate=sampling_rate,
                        metadata={"cache_tag": cache_tag, "cache_key": cache_key},
                        reconstruction=reconstruction_arr,
                        discrete_spikes=discrete_arr,
                    )
                if not loaded:
                    print(f"[eval-only] No cached methods loaded for {summary_dir}; skipping.")
                    continue

                corr_sigma_ms = float(cfg.corr_sigma_ms)

                correlations = compute_correlations_windowed(
                    list(loaded.values()),
                    spike_times,
                    windows,
                    reference_fs=reference_fs,
                    sigma_ms=corr_sigma_ms,
                )
                epochwise_counts = compute_epochwise_counts(list(loaded.values()), spike_times, windows)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(summary, dict):
                    print(f"[eval-only] Failed to read summary json: {summary_path}")
                    continue
                existing_corr = summary.get("correlations")
                if isinstance(existing_corr, dict):
                    merged = dict(existing_corr)
                    merged.update(ensure_serializable(correlations))
                    summary["correlations"] = merged
                else:
                    summary["correlations"] = ensure_serializable(correlations)
                summary["corr_sigma_ms"] = float(corr_sigma_ms)
                summary["epoch_windows_s"] = ensure_serializable(windows)
                summary["epochwise_counts"] = ensure_serializable(epochwise_counts)
                summary["gt_count"] = int(sum(epochwise_counts.get("gt_count", [])))
                for method_name, result in loaded.items():
                    count_key = f"{method_name}_samples"
                    values = epochwise_counts.get(count_key)
                    if values is not None:
                        summary[count_key] = int(sum(values))
                if cfg.trialwise_correlations:
                    trialwise = compute_trialwise_correlations_windowed(
                        list(loaded.values()),
                        spike_times,
                        trial_windows=windows,
                        reference_fs=reference_fs,
                        sigma_ms=corr_sigma_ms,
                    )
                    existing_trialwise = summary.get("trialwise_correlations")
                    if isinstance(existing_trialwise, dict):
                        merged_tw = dict(existing_trialwise)
                        merged_tw.update(ensure_serializable(trialwise))
                        summary["trialwise_correlations"] = merged_tw
                    else:
                        summary["trialwise_correlations"] = ensure_serializable(trialwise)
                    summary["trial_windows_s"] = ensure_serializable(windows)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                summaries.append(summary_path)
                continue

            ds_cfg = DatasetRunConfig(
                dataset_path=dataset_path,
                neuron_type=cfg.neuron_type,
                smoothing=smoothing,
                reference_fs=target,
                edges=edges,
                selection=selection,
                use_cache=cfg.use_cache,
                bm_sigma_gap_s=cfg.bm_sigma_spike_gap,
                corr_sigma_ms=float(cfg.corr_sigma_ms),
                pgas_resample_fs=cfg.pgas_resample_fs,
                cascade_resample_fs=cfg.cascade_resample_fs,
                pgas_fixed_bm_sigma=cfg.pgas_fixed_bm_sigma,
                pgas_bm_sigma_min=cfg.pgas_bm_sigma_min,
                pgas_bm_sigma_max=cfg.pgas_bm_sigma_max,
                pgas_keep_output_dat_files=cfg.pgas_keep_output_dat_files,
                pgas_bm_sigma_use_low_activity_mask=cfg.pgas_bm_sigma_use_low_activity_mask,
                pgas_sigma2_target=cfg.pgas_sigma2_target,
                pgas_sigma2_alpha=cfg.pgas_sigma2_alpha,
                pgas_sigma2_prior_strength=cfg.pgas_sigma2_prior_strength,
                pgas_noise_calibration_scope=cfg.pgas_noise_calibration_scope,
                pgas_noise_calibration_granularity=cfg.pgas_noise_calibration_granularity,
                pgas_noise_calibration_method=cfg.pgas_noise_calibration_method,
                cascade_discretize=bool(cfg.cascade_discretize),
                cascade_model_name=str(cfg.cascade_model_name),
                oasis_g=tuple(cfg.oasis_g),
                oasis_sn=cfg.oasis_sn,
                oasis_b=cfg.oasis_b,
                oasis_b_nonneg=bool(cfg.oasis_b_nonneg),
                oasis_optimize_g=cfg.oasis_optimize_g,
                oasis_penalty=cfg.oasis_penalty,
                oasis_decimate=cfg.oasis_decimate,
                oasis_max_iter=cfg.oasis_max_iter,
                oasis_shift=cfg.oasis_shift,
                oasis_window=cfg.oasis_window,
                oasis_tol=cfg.oasis_tol,
                oasis_uniformity_rtol=cfg.oasis_uniformity_rtol,
                oasis_uniformity_atol=cfg.oasis_uniformity_atol,
                trialwise_correlations=bool(cfg.trialwise_correlations),
                trial_indices=selected_trial_indices,
            )
            outputs = run_inference_for_dataset(
                ds_cfg,
                pgas_constants=cfg.pgas_constants,
                pgas_gparam=cfg.pgas_gparam,
                pgas_output_root=cfg.pgas_output_root,
                ens2_pretrained_root=cfg.ens2_pretrained_root,
                cascade_model_root=cfg.cascade_model_root,
            )
            methods: Dict[str, MethodResult] = outputs["methods"]
            correlations: Dict[str, float] = outputs["correlations"]

            summary_dir.mkdir(parents=True, exist_ok=True)

            np.savez(
                summary_dir / "discrete_spikes.npz",
                **{
                    name: (result.discrete_spikes if result.discrete_spikes is not None else np.array([]))
                    for name, result in methods.items()
                },
            )

            downsample_label = outputs["summary"].get("downsample_target", label)

            summary: Dict[str, object] = {
                "dataset": dataset_tag,
                "smoothing": label,
                "downsample_target": downsample_label,
                "resample_tag": run_tag,
                "correlations": ensure_serializable(correlations),
                "corr_sigma_ms": float(ds_cfg.corr_sigma_ms),
                "methods_run": sorted(methods.keys()),
            }
            extra_summary = outputs.get("summary", {}) if isinstance(outputs, dict) else {}
            if isinstance(extra_summary, dict):
                if extra_summary.get("trialwise_correlations") is not None:
                    summary["trialwise_correlations"] = ensure_serializable(extra_summary.get("trialwise_correlations"))
                if extra_summary.get("trial_windows_s") is not None:
                    summary["trial_windows_s"] = ensure_serializable(extra_summary.get("trial_windows_s"))
                if extra_summary.get("epochwise_counts") is not None:
                    summary["epochwise_counts"] = ensure_serializable(extra_summary.get("epochwise_counts"))
                if extra_summary.get("epoch_windows_s") is not None:
                    summary["epoch_windows_s"] = ensure_serializable(extra_summary.get("epoch_windows_s"))
            epochwise_counts = extra_summary.get("epochwise_counts", {}) if isinstance(extra_summary, dict) else {}

            def summary_sample_count(method_name: str, result: MethodResult) -> int:
                if isinstance(epochwise_counts, dict):
                    values = epochwise_counts.get(f"{method_name}_samples")
                    if values is not None:
                        return int(sum(values))
                return _count_samples(result.discrete_spikes)

            if "pgas" in methods:
                pgas_result = methods["pgas"]
                summary.update(
                    {
                        "pgas_cache": pgas_result.metadata.get("config", {}),
                        "pgas_maxspikes": pgas_result.metadata.get("maxspikes"),
                        "pgas_maxspikes_per_bin": pgas_result.metadata.get("maxspikes_per_bin"),
                        "pgas_input_resample_fs": pgas_result.metadata.get("input_resample_fs"),
                        "pgas_samples": summary_sample_count("pgas", pgas_result),
                    }
                )
            if "ens2" in methods:
                ens2_result = methods["ens2"]
                summary.update(
                    {
                        "ens2_cache": ens2_result.metadata.get("config", {}),
                        "ens2_samples": summary_sample_count("ens2", ens2_result),
                    }
                )
            if "cascade" in methods:
                cascade_result = methods["cascade"]
                summary.update(
                    {
                        "cascade_cache": cascade_result.metadata.get("config", {}),
                        "cascade_input_resample_fs": cascade_result.metadata.get(
                            "input_resample_fs", CASCADE_RESAMPLE_FS
                        ),
                        "cascade_samples": summary_sample_count("cascade", cascade_result),
                    }
                )
            if "oasis" in methods:
                oasis_result = methods["oasis"]
                summary.update(
                    {
                        "oasis_cache": ensure_serializable(
                            oasis_result.metadata.get("config", {})
                        ),
                        "oasis_sampling_rate": float(oasis_result.sampling_rate),
                        "oasis_source_version": oasis_result.metadata.get("source_version"),
                        "oasis_trials": ensure_serializable(
                            oasis_result.metadata.get("trials", [])
                        ),
                        "oasis_samples": summary_sample_count("oasis", oasis_result),
                    }
                )
            summary["gt_count"] = int(outputs.get("summary", {}).get("gt_count", 0))

            with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)

            def method_entry(label: str, result: MethodResult) -> Dict[str, object]:
                meta = result.metadata or {}
                return {
                    "label": label,
                    "method": result.name,
                    "cache_tag": meta.get("cache_tag"),
                    "cache_key": meta.get("cache_key"),
                    "config": ensure_serializable(meta.get("config", {})),
                    "sampling_rate": result.sampling_rate,
                }

            manifest = {
                "run_tag": run_tag,
                "dataset": dataset_tag,
                "smoothing": label,
                "downsample_target": downsample_label,
                "methods": [method_entry(name, result) for name, result in methods.items()],
                "artifacts": {
                    "summary": str(summary_dir / "summary.json"),
                    "discrete_spikes": str(summary_dir / "discrete_spikes.npz"),
                },
            }

            with (summary_dir / "comparison.json").open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)

            summaries.append(summary_dir / "summary.json")
    return summaries


__all__ = ["RunConfig", "run_batch"]
