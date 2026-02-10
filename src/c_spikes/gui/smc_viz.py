from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from c_spikes.gui.data import DataManager, EpochRef
from c_spikes.inference.types import compute_sampling_rate
from c_spikes.model_eval.model_eval import smooth_spike_train


@dataclass(frozen=True)
class PgasCacheEntry:
    run_tag: str
    cache_tag: str
    cache_key: str
    meta_path: Path
    mat_path: Path
    timestamp: str


@dataclass
class BiophysSmcPayload:
    run_tag: str
    cache_tag: str
    cache_key: str
    epoch_id: str
    burnin: int
    n_samples: int
    full_time: np.ndarray
    full_dff: np.ndarray
    full_spikes: np.ndarray
    run_time: np.ndarray
    b_mean: np.ndarray
    b_std: np.ndarray
    c_mean: np.ndarray
    c_std: np.ndarray
    burst_mean: np.ndarray
    burst_std: np.ndarray
    s_mean: np.ndarray
    s_std: np.ndarray
    bc_mean: np.ndarray
    gt_smooth_time: np.ndarray
    gt_smooth: np.ndarray
    param_names: List[str]
    param_values: np.ndarray


def list_spike_inference_runs(data_dir: Path) -> List[str]:
    parent = Path(data_dir) / "spike_inference"
    if not parent.exists():
        return []
    runs = [p.name for p in parent.iterdir() if p.is_dir()]
    return sorted(runs, key=_natural_sort_key)


def list_pgas_cache_entries(data_dir: Path, run_tag: str) -> List[PgasCacheEntry]:
    pgas_root = Path(data_dir) / "spike_inference" / run_tag / "inference_cache" / "pgas"
    if not pgas_root.exists():
        return []
    by_tag: Dict[str, Tuple[float, PgasCacheEntry]] = {}
    for cache_dir in sorted(pgas_root.iterdir()):
        if not cache_dir.is_dir():
            continue
        for meta_path in sorted(cache_dir.glob("*.json")):
            mat_path = meta_path.with_suffix(".mat")
            if not mat_path.exists():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            cache_tag = str(
                metadata.get("cache_tag")
                or payload.get("dataset")
                or cache_dir.name
            ).strip()
            if not cache_tag:
                cache_tag = cache_dir.name
            cache_key = str(payload.get("cache_key") or meta_path.stem).strip() or meta_path.stem
            timestamp = str(payload.get("timestamp") or "")
            mtime = max(meta_path.stat().st_mtime, mat_path.stat().st_mtime)
            entry = PgasCacheEntry(
                run_tag=run_tag,
                cache_tag=cache_tag,
                cache_key=cache_key,
                meta_path=meta_path,
                mat_path=mat_path,
                timestamp=timestamp,
            )
            previous = by_tag.get(cache_tag)
            if previous is None or mtime > previous[0]:
                by_tag[cache_tag] = (mtime, entry)
    return [v[1] for v in sorted(by_tag.values(), key=lambda pair: _natural_sort_key(pair[1].cache_tag))]


def cache_tag_to_epoch_id(cache_tag: str) -> str:
    token = str(cache_tag).strip()
    match = re.search(r"(.+_epoch\d+)", token)
    if match:
        return str(match.group(1))
    split = re.split(r"_s[^_]*_ms\d+", token, maxsplit=1)
    if split and split[0]:
        return split[0]
    return token


def load_biophys_smc_payload(
    *,
    data_dir: Path,
    entry: PgasCacheEntry,
    data_manager: DataManager,
    epoch_refs: List[EpochRef],
) -> BiophysSmcPayload:
    meta = json.loads(entry.meta_path.read_text(encoding="utf-8"))
    metadata = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    burnin = int(metadata.get("burnin", 100))
    cache_tag = entry.cache_tag
    epoch_id = cache_tag_to_epoch_id(cache_tag)
    epoch = _epoch_by_id(epoch_refs).get(epoch_id)
    if epoch is None:
        raise ValueError(f"Could not resolve epoch from cache tag '{cache_tag}'.")
    full_time, full_dff, full_spikes_opt = data_manager.load_epoch(epoch)
    full_spikes = np.asarray(
        [] if full_spikes_opt is None else full_spikes_opt,
        dtype=np.float64,
    ).ravel()

    fallback_output_root = Path(data_dir) / "spike_inference" / entry.run_tag / "pgas_output"
    output_root_raw = str(metadata.get("output_root") or "").strip()
    output_root = Path(output_root_raw) if output_root_raw else fallback_output_root
    if not output_root.exists():
        output_root = fallback_output_root
    traj_file = _pick_latest(output_root, f"traj_samples_{cache_tag}_trial*.dat")
    param_file = _pick_latest(output_root, f"param_samples_{cache_tag}_trial*.dat")
    if traj_file is None:
        raise FileNotFoundError(f"No trajectory file found for cache tag '{cache_tag}' in {output_root}")
    if param_file is None:
        raise FileNotFoundError(f"No parameter trace file found for cache tag '{cache_tag}' in {output_root}")

    traj = _load_traj_stats(traj_file, burnin=burnin)
    mat_data = sio.loadmat(entry.mat_path, squeeze_me=True)
    run_time = np.asarray(mat_data.get("time_stamps", []), dtype=np.float64).ravel()
    if run_time.size == 0:
        run_time = np.asarray(full_time, dtype=np.float64).ravel()

    n = min(
        run_time.size,
        traj["b_mean"].size,
        traj["c_mean"].size,
        traj["burst_mean"].size,
        traj["s_mean"].size,
    )
    if n <= 0:
        raise ValueError(f"No aligned trajectory samples for cache tag '{cache_tag}'.")
    run_time = run_time[:n]
    b_mean = traj["b_mean"][:n]
    b_std = traj["b_std"][:n]
    c_mean = traj["c_mean"][:n]
    c_std = traj["c_std"][:n]
    burst_mean = traj["burst_mean"][:n]
    burst_std = traj["burst_std"][:n]
    s_mean = traj["s_mean"][:n]
    s_std = traj["s_std"][:n]
    bc_mean = b_mean + c_mean

    gt_smooth_time, gt_smooth = _smooth_ground_truth(full_time, full_spikes)
    param_names, param_values = _load_param_traces(param_file)

    return BiophysSmcPayload(
        run_tag=entry.run_tag,
        cache_tag=cache_tag,
        cache_key=entry.cache_key,
        epoch_id=epoch_id,
        burnin=traj["burnin_eff"],
        n_samples=traj["n_samples"],
        full_time=np.asarray(full_time, dtype=np.float64).ravel(),
        full_dff=np.asarray(full_dff, dtype=np.float64).ravel(),
        full_spikes=full_spikes,
        run_time=run_time,
        b_mean=b_mean,
        b_std=b_std,
        c_mean=c_mean,
        c_std=c_std,
        burst_mean=burst_mean,
        burst_std=burst_std,
        s_mean=s_mean,
        s_std=s_std,
        bc_mean=bc_mean,
        gt_smooth_time=gt_smooth_time,
        gt_smooth=gt_smooth,
        param_names=param_names,
        param_values=param_values,
    )


def _natural_sort_key(token: str) -> tuple:
    parts = re.split(r"(\d+)", str(token))
    out: List[object] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return tuple(out)


def _epoch_by_id(epoch_refs: List[EpochRef]) -> Dict[str, EpochRef]:
    return {ref.epoch_id: ref for ref in epoch_refs}


def _pick_latest(root: Path, pattern: str) -> Optional[Path]:
    try:
        matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    except Exception:
        return None
    if not matches:
        return None
    return matches[-1]


def _load_traj_stats(path: Path, *, burnin: int) -> Dict[str, np.ndarray | int]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = np.asarray([data], dtype=float)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"Unexpected trajectory shape in {path}: {data.shape}")
    idx = np.asarray(data[:, 0], dtype=np.float64)
    burst = np.asarray(data[:, 1], dtype=np.float64)
    b = np.asarray(data[:, 2], dtype=np.float64)
    s = np.asarray(data[:, 3], dtype=np.float64)
    c = np.asarray(data[:, 4], dtype=np.float64)

    time_len = int(np.sum(idx == 0))
    if time_len <= 0:
        raise ValueError(f"Could not infer PGAS trajectory TIME from {path}")
    n_samples = int(data.shape[0] // time_len)
    if n_samples <= 0:
        raise ValueError(f"No PGAS samples found in {path}")
    n_rows = n_samples * time_len
    burst_mat = burst[:n_rows].reshape((n_samples, time_len))
    b_mat = b[:n_rows].reshape((n_samples, time_len))
    s_mat = s[:n_rows].reshape((n_samples, time_len))
    c_mat = c[:n_rows].reshape((n_samples, time_len))

    burnin_eff = int(max(0, min(int(burnin), n_samples - 1)))
    sample_slice = slice(burnin_eff, None)
    return {
        "n_samples": n_samples,
        "burnin_eff": burnin_eff,
        "burst_mean": np.mean(burst_mat[sample_slice], axis=0),
        "burst_std": np.std(burst_mat[sample_slice], axis=0),
        "b_mean": np.mean(b_mat[sample_slice], axis=0),
        "b_std": np.std(b_mat[sample_slice], axis=0),
        "s_mean": np.mean(s_mat[sample_slice], axis=0),
        "s_std": np.std(s_mat[sample_slice], axis=0),
        "c_mean": np.mean(c_mat[sample_slice], axis=0),
        "c_std": np.std(c_mat[sample_slice], axis=0),
    }


def _load_param_traces(path: Path) -> Tuple[List[str], np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        return [], np.zeros((0, 0), dtype=np.float64)
    if data.dtype.names is None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        names = [f"param_{i + 1}" for i in range(arr.shape[1])]
        return names, arr
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    names = [str(name) for name in data.dtype.names]
    cols = [np.asarray(data[name], dtype=np.float64).ravel() for name in names]
    values = np.column_stack(cols) if cols else np.zeros((0, 0), dtype=np.float64)
    return names, values


def _smooth_ground_truth(time: np.ndarray, spike_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=np.float64).ravel()
    if t.size < 2:
        return t, np.zeros_like(t)
    spikes = np.asarray(spike_times, dtype=np.float64).ravel()
    spikes = spikes[np.isfinite(spikes)]
    t0 = float(t[0])
    duration = float(max(t[-1] - t0, 0.0))
    fs = float(compute_sampling_rate(t))
    spikes_rel = spikes - t0
    spikes_rel = spikes_rel[(spikes_rel >= 0.0) & (spikes_rel <= duration)]
    smoothed = smooth_spike_train(spikes_rel, fs, duration=duration, sigma_ms=20)
    smooth_time = t0 + np.arange(smoothed.size, dtype=np.float64) / fs
    return smooth_time, np.asarray(smoothed, dtype=np.float64)
