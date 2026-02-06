from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from c_spikes.utils import load_Janelia_data

from .cache import CACHE_ROOT, save_method_cache
from .smoothing import mean_downsample_trace
from .types import (
    MethodResult,
    TrialSeries,
    compute_config_signature,
    compute_sampling_rate,
    ensure_serializable,
    flatten_trials,
    hash_series,
)


TRIALWISE_CORRELATION_FIELDNAMES: Tuple[str, ...] = (
    "run",
    "dataset",
    "smoothing",
    "method",
    "label",
    "corr_sigma_ms",
    "trial",
    "start_s",
    "end_s",
    "correlation",
)


def _parse_smoothing_to_target_fs(label: str) -> Optional[float]:
    token = str(label).strip().lower()
    if token in {"raw", ""}:
        return None
    if token in {"30hz", "30"}:
        return 30.0
    if token in {"10hz", "10"}:
        return 10.0
    if token.endswith("hz"):
        try:
            return float(token[:-2])
        except Exception:
            return None
    return None


def compute_trace_hash_for_dataset(
    dataset_path: Path,
    *,
    smoothing: str,
) -> str:
    """
    Compute a stable trace hash for a dataset/smoothing label.

    This is intended for creating external cache entries that can still participate in the
    existing inference cache metadata validation.
    """
    dataset_path = dataset_path.expanduser().resolve()
    time_stamps, dff, _spike_times = load_Janelia_data(str(dataset_path))
    target_fs = _parse_smoothing_to_target_fs(smoothing)

    trials: List[TrialSeries] = []
    for idx in range(int(time_stamps.shape[0])):
        t = np.asarray(time_stamps[idx], dtype=np.float64).ravel()
        y = np.asarray(dff[idx], dtype=np.float64).ravel()
        m = np.isfinite(t) & np.isfinite(y)
        t = t[m]
        y = y[m]
        if t.size == 0:
            continue
        if target_fs is not None:
            ds = mean_downsample_trace(t, y, float(target_fs))
            t = np.asarray(ds.times, dtype=np.float64).ravel()
            y = np.asarray(ds.values, dtype=np.float64).ravel()
        trials.append(TrialSeries(times=t, values=y))
    if not trials:
        raise RuntimeError(f"No valid trials found in {dataset_path}")
    times, values = flatten_trials(trials)
    return hash_series(times, values)


def _load_array_dict(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return dict(sio.loadmat(path))
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    raise ValueError(f"Unsupported prediction file type: {path} (expected .mat or .npz)")


def _as_float_array(name: str, value: Any) -> np.ndarray:
    if value is None:
        raise KeyError(f"Missing key {name!r} in prediction file.")
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise TypeError(f"{name!r} must be real-valued; got complex dtype {arr.dtype}")
    return np.asarray(arr, dtype=np.float64)


def _flatten_trial_matrix(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if times.ndim == 1 and values.ndim == 1:
        t = times.ravel()
        y = values.ravel()
        if t.shape != y.shape:
            raise ValueError(f"time/value length mismatch: {t.shape} vs {y.shape}")
        m = np.isfinite(t)
        t = t[m]
        y = y[m]
        return t, y

    if times.ndim == 1 and values.ndim == 2:
        shared_t = times.ravel()
        trials: List[TrialSeries] = []
        for i in range(int(values.shape[0])):
            y = values[i].ravel()
            if y.shape != shared_t.shape:
                raise ValueError(f"time/value shape mismatch for trial {i}: {shared_t.shape} vs {y.shape}")
            m = np.isfinite(shared_t)
            t_i = shared_t[m]
            y_i = y[m]
            if t_i.size:
                trials.append(TrialSeries(times=t_i, values=y_i))
        if not trials:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        return flatten_trials(trials)

    if times.ndim == 2 and values.ndim == 2:
        if times.shape != values.shape:
            raise ValueError(f"time/value shape mismatch: {times.shape} vs {values.shape}")
        trials = []
        for i in range(int(times.shape[0])):
            t = times[i].ravel()
            y = values[i].ravel()
            m = np.isfinite(t)
            t = t[m]
            y = y[m]
            if t.size:
                trials.append(TrialSeries(times=t, values=y))
        if not trials:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        return flatten_trials(trials)

    raise ValueError(
        f"Unsupported shapes for time/value: time {times.shape} (ndim={times.ndim}), "
        f"value {values.shape} (ndim={values.ndim}). Expected 1D/1D or 2D/2D or 1D/2D."
    )


def load_method_result_from_file(
    pred_path: Path,
    *,
    method: str,
    time_key: str = "time_stamps",
    spike_prob_key: str = "spike_prob",
    reconstruction_key: Optional[str] = "reconstruction",
    discrete_spikes_key: Optional[str] = "discrete_spikes",
) -> MethodResult:
    data = _load_array_dict(pred_path)
    times = _as_float_array(time_key, data.get(time_key))
    prob = _as_float_array(spike_prob_key, data.get(spike_prob_key))

    t_flat, y_flat = _flatten_trial_matrix(times.squeeze(), prob.squeeze())
    if t_flat.size < 2:
        raise ValueError(f"Imported time_stamps has insufficient finite samples: {pred_path}")

    fs = compute_sampling_rate(np.asarray(t_flat, dtype=np.float64).ravel())

    reconstruction = None
    if reconstruction_key:
        raw = data.get(reconstruction_key)
        if raw is not None:
            rec = _as_float_array(reconstruction_key, raw)
            t_rec, rec_flat = _flatten_trial_matrix(times.squeeze(), rec.squeeze())
            if t_rec.shape != t_flat.shape or not np.allclose(t_rec, t_flat, atol=0.0, rtol=0.0):
                raise ValueError(
                    f"{reconstruction_key!r} time axis does not match {time_key!r} after flattening; "
                    "pass --reconstruction-key '' to ignore."
                )
            reconstruction = np.asarray(rec_flat, dtype=np.float64)
    discrete = None
    if discrete_spikes_key:
        raw = data.get(discrete_spikes_key)
        if raw is not None:
            disc = _as_float_array(discrete_spikes_key, raw)
            t_disc, disc_flat = _flatten_trial_matrix(times.squeeze(), disc.squeeze())
            if t_disc.shape != t_flat.shape or not np.allclose(t_disc, t_flat, atol=0.0, rtol=0.0):
                raise ValueError(
                    f"{discrete_spikes_key!r} time axis does not match {time_key!r} after flattening; "
                    "pass --discrete-spikes-key '' to ignore."
                )
            discrete = np.asarray(disc_flat, dtype=np.float64)

    return MethodResult(
        name=str(method),
        time_stamps=np.asarray(t_flat, dtype=np.float64),
        spike_prob=np.asarray(y_flat, dtype=np.float64),
        sampling_rate=float(fs),
        metadata={"source": str(pred_path)},
        reconstruction=reconstruction,
        discrete_spikes=discrete,
    )


def write_full_evaluation_stub(
    *,
    eval_root: Path,
    run_tag: str,
    dataset: str,
    smoothing: str,
    method: str,
    label: str,
    cache_tag: str,
    cache_key: str,
    config: Mapping[str, Any],
    sampling_rate: float,
    corr_sigma_ms: float = 50.0,
) -> Tuple[Path, Path]:
    """
    Create/update `results/full_evaluation/<run>/<dataset>/<smoothing>/{comparison,summary}.json`.
    """
    eval_root = eval_root.expanduser().resolve()
    summary_dir = eval_root / str(run_tag) / str(dataset) / str(smoothing)
    summary_dir.mkdir(parents=True, exist_ok=True)

    cmp_path = summary_dir / "comparison.json"
    if cmp_path.exists():
        try:
            obj = json.loads(cmp_path.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
    else:
        obj = {}

    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("run_tag", str(run_tag))
    obj.setdefault("dataset", str(dataset))
    obj.setdefault("smoothing", str(smoothing))
    obj.setdefault("downsample_target", str(smoothing))

    methods = obj.get("methods")
    if not isinstance(methods, list):
        methods = []

    config_hash, config_ser = compute_config_signature(dict(config))
    if config_hash != str(cache_key):
        raise ValueError(f"cache_key mismatch: computed {config_hash} but caller provided {cache_key}")

    new_entry: Dict[str, Any] = {
        "label": str(label),
        "method": str(method),
        "cache_tag": str(cache_tag),
        "cache_key": str(cache_key),
        "config": ensure_serializable(config_ser),
        "sampling_rate": float(sampling_rate),
    }

    replaced = False
    for idx, entry in enumerate(list(methods)):
        if isinstance(entry, dict) and str(entry.get("method", "")).strip() == str(method):
            methods[idx] = new_entry
            replaced = True
            break
    if not replaced:
        methods.append(new_entry)
    obj["methods"] = methods

    cmp_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    summary_path = summary_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    else:
        summary = {}
    if not isinstance(summary, dict):
        summary = {}
    summary.setdefault("dataset", str(dataset))
    summary.setdefault("smoothing", str(smoothing))
    summary.setdefault("downsample_target", str(smoothing))
    summary.setdefault("resample_tag", str(run_tag))
    summary.setdefault("corr_sigma_ms", float(corr_sigma_ms))
    methods_run = summary.get("methods_run")
    if not isinstance(methods_run, list):
        methods_run = []
    if str(method) not in methods_run:
        methods_run.append(str(method))
    summary["methods_run"] = methods_run
    summary.setdefault("correlations", {})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return cmp_path, summary_path


def import_external_method(
    *,
    pred_path: Path,
    method: str,
    dataset: str,
    smoothing: str,
    run_tag: str,
    data_root: Optional[Path] = None,
    eval_root: Path = Path("results/full_evaluation"),
    cache_root: Path = CACHE_ROOT,
    cache_tag: Optional[str] = None,
    label: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    time_key: str = "time_stamps",
    spike_prob_key: str = "spike_prob",
    reconstruction_key: Optional[str] = "reconstruction",
    discrete_spikes_key: Optional[str] = "discrete_spikes",
    corr_sigma_ms: float = 50.0,
) -> Dict[str, Any]:
    """
    Import an externally generated method output into:
      - `results/inference_cache/<method>/<cache_tag>/<cache_key>.mat|.json`
      - `results/full_evaluation/<run>/<dataset>/<smoothing>/{comparison,summary}.json`
    """
    method = str(method).strip()
    dataset = str(dataset).strip()
    smoothing = str(smoothing).strip()
    run_tag = str(run_tag).strip()
    if not method:
        raise ValueError("method must be non-empty.")
    if not dataset:
        raise ValueError("dataset must be non-empty.")
    if not smoothing:
        raise ValueError("smoothing must be non-empty.")
    if not run_tag:
        raise ValueError("run_tag must be non-empty.")

    pred_path = pred_path.expanduser().resolve()
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    if cache_tag is None:
        cache_tag = dataset
    cache_tag = str(cache_tag).strip()
    if not cache_tag:
        raise ValueError("cache_tag must be non-empty.")

    if label is None:
        label = method
    label = str(label).strip() or method

    if config is None:
        config = {"source": "external", "pred_file": pred_path.name}
    config_dict = dict(config)
    config_hash, _config_ser = compute_config_signature(config_dict)

    trace_hash = "external"
    dataset_path = None
    if data_root is not None:
        data_root = data_root.expanduser().resolve()
        dataset_path = data_root / f"{dataset}.mat"
        if dataset_path.exists():
            trace_hash = compute_trace_hash_for_dataset(dataset_path, smoothing=smoothing)

    result = load_method_result_from_file(
        pred_path,
        method=method,
        time_key=time_key,
        spike_prob_key=spike_prob_key,
        reconstruction_key=reconstruction_key,
        discrete_spikes_key=discrete_spikes_key,
    )
    save_method_cache(
        method,
        cache_tag,
        result,
        config_dict,
        trace_hash,
        cache_root=cache_root,
    )

    comparison_path, summary_path = write_full_evaluation_stub(
        eval_root=eval_root,
        run_tag=run_tag,
        dataset=dataset,
        smoothing=smoothing,
        method=method,
        label=label,
        cache_tag=cache_tag,
        cache_key=config_hash,
        config=config_dict,
        sampling_rate=float(result.sampling_rate),
        corr_sigma_ms=float(corr_sigma_ms),
    )

    return {
        "method": method,
        "dataset": dataset,
        "smoothing": smoothing,
        "run_tag": run_tag,
        "pred_path": str(pred_path),
        "dataset_path": (str(dataset_path) if dataset_path is not None else None),
        "trace_hash": trace_hash,
        "cache_tag": cache_tag,
        "cache_key": config_hash,
        "comparison_json": str(comparison_path),
        "summary_json": str(summary_path),
        "sampling_rate": float(result.sampling_rate),
    }


def merge_trialwise_correlations_csv(
    base_csv: Path,
    incoming_csv: Path,
    *,
    out_csv: Optional[Path] = None,
    on_conflict: str = "replace",
) -> Path:
    """
    Merge an external trialwise correlation CSV (same schema as results/trialwise_correlations.csv).

    Conflict key: (run, dataset, smoothing, method, corr_sigma_ms, trial).
    """
    on_conflict = str(on_conflict).strip().lower()
    if on_conflict not in {"replace", "keep"}:
        raise ValueError("on_conflict must be 'replace' or 'keep'")

    base_csv = base_csv.expanduser().resolve()
    incoming_csv = incoming_csv.expanduser().resolve()
    if out_csv is None:
        out_csv = base_csv
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def _read_rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = [dict(r) for r in reader]
        return rows

    base_rows = _read_rows(base_csv)
    new_rows = _read_rows(incoming_csv)

    def _validate(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
        if not rows:
            return
        missing = [k for k in TRIALWISE_CORRELATION_FIELDNAMES if k not in rows[0]]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

    _validate(base_rows, base_csv)
    _validate(new_rows, incoming_csv)

    def _key(row: Mapping[str, Any]) -> Tuple[str, str, str, str, str, str]:
        return (
            str(row.get("run", "")).strip(),
            str(row.get("dataset", "")).strip(),
            str(row.get("smoothing", "")).strip(),
            str(row.get("method", "")).strip(),
            str(row.get("corr_sigma_ms", "")).strip(),
            str(row.get("trial", "")).strip(),
        )

    merged: Dict[Tuple[str, str, str, str, str, str], Dict[str, str]] = {}
    for row in base_rows:
        merged[_key(row)] = {k: str(row.get(k, "")) for k in TRIALWISE_CORRELATION_FIELDNAMES}
    for row in new_rows:
        k = _key(row)
        if k in merged and on_conflict == "keep":
            continue
        merged[k] = {k2: str(row.get(k2, "")) for k2 in TRIALWISE_CORRELATION_FIELDNAMES}

    rows_out = list(merged.values())
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRIALWISE_CORRELATION_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows_out)
    return out_csv
