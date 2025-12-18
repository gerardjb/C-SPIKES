"""
Reusable batching/orchestration layer for running spike-inference methods.

This wraps the existing compare_inference_methods helpers so callers can:
  • select any subset of methods (pgas / ens2 / cascade),
  • batch over datasets via globbing or explicit lists,
  • drive runs either from Python (direct call) or the CLI wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from c_spikes.inference.cascade import CASCADE_RESAMPLE_FS
from c_spikes.inference.pgas import PGAS_BM_SIGMA_DEFAULT
from c_spikes.inference.smoothing import resolve_smoothing_levels
from c_spikes.inference.types import MethodResult, ensure_serializable
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
    bm_sigma_spike_gap: float = 0.15
    pgas_constants: Path = Path("parameter_files/constants_GCaMP8_soma.json")
    pgas_gparam: Path = Path("src/c_spikes/pgas/20230525_gold.dat")
    pgas_output_root: Path = Path("results/pgas_output/comparison")
    pgas_resample_fs: Optional[float] = None
    cascade_resample_fs: Optional[float] = None  # None => use input sampling rate (no forced resample)
    cascade_discretize: bool = True
    pgas_maxspikes: Optional[int] = None
    pgas_fixed_bm_sigma: Optional[float] = PGAS_BM_SIGMA_DEFAULT
    run_tag: Optional[str] = None  # optional override
    pgas_c0_first_y: bool = False
    trialwise_correlations: bool = False


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

    summaries: List[Path] = []
    for dataset_path in dataset_paths:
        dataset_tag = dataset_path.stem
        for label, target in smoothing_levels:
            selection = MethodSelection(
                run_pgas=("pgas" in method_list),
                run_ens2=("ens2" in method_list),
                run_cascade=("cascade" in method_list),
            )
            edges = None
            if cfg.edges_path.exists():
                edges_lookup = np.load(cfg.edges_path, allow_pickle=True).item()
                if dataset_tag in edges_lookup:
                    edges = np.asarray(edges_lookup[dataset_tag], dtype=np.float64)
            smoothing = SmoothingLevel(label=label, target_fs=target)
            ds_cfg = DatasetRunConfig(
                dataset_path=dataset_path,
                neuron_type=cfg.neuron_type,
                smoothing=smoothing,
                reference_fs=target,
                edges=edges,
                selection=selection,
                use_cache=cfg.use_cache,
                bm_sigma_gap_s=cfg.bm_sigma_spike_gap,
                pgas_resample_fs=cfg.pgas_resample_fs,
                cascade_resample_fs=cfg.cascade_resample_fs,
                pgas_fixed_bm_sigma=cfg.pgas_fixed_bm_sigma,
                cascade_discretize=bool(cfg.cascade_discretize),
                trialwise_correlations=bool(cfg.trialwise_correlations),
            )
            outputs = run_inference_for_dataset(
                ds_cfg,
                pgas_constants=cfg.pgas_constants,
                pgas_gparam=cfg.pgas_gparam,
                pgas_output_root=cfg.pgas_output_root,
                ens2_pretrained_root=Path("results/Pretrained_models/ens2_published"),
                cascade_model_root=Path("results/Pretrained_models"),
            )
            methods: Dict[str, MethodResult] = outputs["methods"]
            correlations: Dict[str, float] = outputs["correlations"]

            summary_dir = cfg.output_root / run_tag / dataset_tag / label
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
                "methods_run": sorted(methods.keys()),
            }
            extra_summary = outputs.get("summary", {}) if isinstance(outputs, dict) else {}
            if isinstance(extra_summary, dict):
                if extra_summary.get("trialwise_correlations") is not None:
                    summary["trialwise_correlations"] = ensure_serializable(extra_summary.get("trialwise_correlations"))
                if extra_summary.get("trial_windows_s") is not None:
                    summary["trial_windows_s"] = ensure_serializable(extra_summary.get("trial_windows_s"))
            if "pgas" in methods:
                pgas_result = methods["pgas"]
                summary.update(
                    {
                        "pgas_cache": pgas_result.metadata.get("config", {}),
                        "pgas_maxspikes": pgas_result.metadata.get("maxspikes"),
                        "pgas_maxspikes_per_bin": pgas_result.metadata.get("maxspikes_per_bin"),
                        "pgas_input_resample_fs": pgas_result.metadata.get("input_resample_fs"),
                        "pgas_samples": _count_samples(pgas_result.discrete_spikes),
                    }
                )
            if "ens2" in methods:
                ens2_result = methods["ens2"]
                summary.update(
                    {
                        "ens2_cache": ens2_result.metadata.get("config", {}),
                        "ens2_samples": _count_samples(ens2_result.discrete_spikes),
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
                        "cascade_samples": _count_samples(cascade_result.discrete_spikes),
                    }
                )
            summary["gt_count"] = int(outputs.get("summary", {}).get("gt_count", 0))

            with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
                import json

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
                import json

                json.dump(manifest, fh, indent=2)

            summaries.append(summary_dir / "summary.json")
    return summaries


__all__ = ["RunConfig", "run_batch"]
