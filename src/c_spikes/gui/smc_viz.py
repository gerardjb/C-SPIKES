from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    trial_index: Optional[int] = None
    epoch_id_hint: Optional[str] = None
    display_tag: Optional[str] = None


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


def list_pgas_cache_entries(
    data_dir: Path,
    run_tag: str,
    *,
    epoch_refs: Optional[Sequence[EpochRef]] = None,
) -> List[PgasCacheEntry]:
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
    base_entries = [v[1] for v in sorted(by_tag.values(), key=lambda pair: _natural_sort_key(pair[1].cache_tag))]
    expanded: List[PgasCacheEntry] = []
    refs = list(epoch_refs or [])
    for entry in base_entries:
        expanded.extend(_expand_entry_for_trials(data_dir=Path(data_dir), entry=entry, epoch_refs=refs))
    return sorted(expanded, key=lambda e: _natural_sort_key(e.display_tag or e.cache_tag))


def cache_tag_to_epoch_id(cache_tag: str) -> str:
    token = str(cache_tag).strip()
    explicit = _extract_epoch_id_from_tag(token)
    if explicit:
        return explicit
    split = re.split(r"_s[^_]*_ms\d+", token, maxsplit=1)
    if split and split[0]:
        return split[0]
    return token


def cache_tag_to_dataset_stem(cache_tag: str) -> str:
    epoch_id = cache_tag_to_epoch_id(cache_tag)
    return re.sub(r"_epoch\d+$", "", epoch_id)


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
    epoch_id = str(entry.epoch_id_hint or cache_tag_to_epoch_id(cache_tag))
    epoch_lookup = _epoch_by_id(epoch_refs)
    epoch = epoch_lookup.get(epoch_id)
    if epoch is None and entry.trial_index is not None:
        dataset_stem = cache_tag_to_dataset_stem(cache_tag)
        original_idx = int(entry.trial_index)
        selected = _trial_indices_from_metadata(metadata)
        if selected is None:
            selected = _run_trial_selection_lookup(data_dir, entry.run_tag).get(dataset_stem)
        if selected is not None and entry.trial_index < len(selected):
            original_idx = int(selected[entry.trial_index])
        guessed = _epoch_id_for_dataset_trial(dataset_stem, original_idx, epoch_refs)
        if guessed is not None:
            epoch_id = guessed
            epoch = epoch_lookup.get(epoch_id)
    if epoch is None:
        single_epoch = _single_epoch_id_for_dataset(cache_tag_to_dataset_stem(cache_tag), epoch_refs)
        if single_epoch is not None:
            epoch_id = single_epoch
            epoch = epoch_lookup.get(epoch_id)
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
    traj_file = _pick_trial_file(
        output_root,
        prefix="traj_samples",
        cache_tag=cache_tag,
        trial_index=entry.trial_index,
    )
    param_file = _pick_trial_file(
        output_root,
        prefix="param_samples",
        cache_tag=cache_tag,
        trial_index=entry.trial_index,
    )
    if traj_file is None:
        trial_suffix = "" if entry.trial_index is None else f" (trial {entry.trial_index})"
        raise FileNotFoundError(
            f"No trajectory file found for cache tag '{cache_tag}'{trial_suffix} in {output_root}"
        )
    if param_file is None:
        trial_suffix = "" if entry.trial_index is None else f" (trial {entry.trial_index})"
        raise FileNotFoundError(
            f"No parameter trace file found for cache tag '{cache_tag}'{trial_suffix} in {output_root}"
        )

    traj = _load_traj_stats(traj_file, burnin=burnin)
    mat_data = sio.loadmat(entry.mat_path, squeeze_me=True)
    run_time = np.asarray(mat_data.get("time_stamps", []), dtype=np.float64).ravel()
    if run_time.size == 0:
        run_time = np.asarray(full_time, dtype=np.float64).ravel()
    elif entry.trial_index is not None:
        run_time = _slice_run_time_for_trial(
            run_time=run_time,
            output_root=output_root,
            cache_tag=cache_tag,
            trial_index=int(entry.trial_index),
            expected_len=int(traj["time_len"]),
        )

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


def _extract_epoch_id_from_tag(cache_tag: str) -> Optional[str]:
    token = str(cache_tag).strip()
    match = re.search(r"(.+_epoch\d+)", token)
    if match:
        return str(match.group(1))
    return None


def _load_entry_metadata(entry: PgasCacheEntry) -> Dict[str, object]:
    try:
        payload = json.loads(entry.meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _normalize_trial_indices(value: object) -> Optional[List[int]]:
    if not isinstance(value, list):
        return None
    out: List[int] = []
    seen: set[int] = set()
    for item in value:
        try:
            idx = int(item)
        except Exception:
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return sorted(out)


def _trial_indices_from_metadata(metadata: Dict[str, object]) -> Optional[List[int]]:
    config = metadata.get("config")
    if isinstance(config, dict):
        parsed = _normalize_trial_indices(config.get("trial_indices"))
        if parsed:
            return parsed
    parsed = _normalize_trial_indices(metadata.get("trial_indices"))
    if parsed:
        return parsed
    return None


def _run_trial_selection_lookup(data_dir: Path, run_tag: str) -> Dict[str, List[int]]:
    path = Path(data_dir) / "spike_inference" / run_tag / "slurm" / "trial_selection.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for key, value in payload.items():
        dataset = str(key).strip()
        if not dataset:
            continue
        parsed = _normalize_trial_indices(value)
        if parsed:
            out[dataset] = parsed
    return out


def _entry_output_root(data_dir: Path, entry: PgasCacheEntry, metadata: Dict[str, object]) -> Path:
    fallback = Path(data_dir) / "spike_inference" / entry.run_tag / "pgas_output"
    raw = str(metadata.get("output_root") or "").strip()
    output_root = Path(raw) if raw else fallback
    if not output_root.exists():
        output_root = fallback
    return output_root


def _parse_trial_index_from_name(name: str) -> Optional[int]:
    match = re.search(r"_trial(\d+)(?:\D.*)?\.dat$", str(name))
    if not match:
        return None
    return int(match.group(1))


def _trial_indices_from_output(output_root: Path, cache_tag: str) -> List[int]:
    out: set[int] = set()
    try:
        paths = list(output_root.glob(f"traj_samples_{cache_tag}_trial*.dat"))
    except Exception:
        return []
    for path in paths:
        idx = _parse_trial_index_from_name(path.name)
        if idx is not None:
            out.add(idx)
    return sorted(out)


def _single_epoch_id_for_dataset(dataset_stem: str, epoch_refs: Sequence[EpochRef]) -> Optional[str]:
    matches = [ref for ref in epoch_refs if ref.file_path.stem == dataset_stem]
    if len(matches) == 1:
        return matches[0].epoch_id
    return None


def _epoch_id_for_dataset_trial(
    dataset_stem: str,
    original_trial_index: int,
    epoch_refs: Sequence[EpochRef],
) -> Optional[str]:
    expected = f"{dataset_stem}_epoch{int(original_trial_index) + 1}"
    by_id = _epoch_by_id(list(epoch_refs))
    if expected in by_id:
        return expected
    for ref in epoch_refs:
        if ref.file_path.stem == dataset_stem and ref.epoch_index == int(original_trial_index):
            return ref.epoch_id
    return None


def _expand_entry_for_trials(
    *,
    data_dir: Path,
    entry: PgasCacheEntry,
    epoch_refs: Sequence[EpochRef],
) -> List[PgasCacheEntry]:
    explicit_epoch = _extract_epoch_id_from_tag(entry.cache_tag)
    if explicit_epoch is not None:
        return [replace(entry, epoch_id_hint=explicit_epoch, display_tag=entry.cache_tag)]

    metadata = _load_entry_metadata(entry)
    output_root = _entry_output_root(data_dir, entry, metadata)
    trial_indices = _trial_indices_from_output(output_root, entry.cache_tag)
    dataset_stem = cache_tag_to_dataset_stem(entry.cache_tag)

    if not trial_indices:
        single_epoch = _single_epoch_id_for_dataset(dataset_stem, epoch_refs)
        label = entry.cache_tag if single_epoch is None else f"{entry.cache_tag} | {single_epoch}"
        return [replace(entry, epoch_id_hint=single_epoch, display_tag=label)]

    selected = _trial_indices_from_metadata(metadata)
    if selected is None:
        selected = _run_trial_selection_lookup(data_dir, entry.run_tag).get(dataset_stem)

    expanded: List[PgasCacheEntry] = []
    for local_idx in trial_indices:
        original_idx = int(local_idx)
        if selected is not None and local_idx < len(selected):
            original_idx = int(selected[local_idx])
        epoch_id = _epoch_id_for_dataset_trial(dataset_stem, original_idx, epoch_refs)
        if epoch_id is None:
            label = f"{entry.cache_tag} | trial{local_idx}"
        else:
            label = f"{entry.cache_tag} | {epoch_id} [trial{local_idx}]"
        expanded.append(
            replace(
                entry,
                trial_index=int(local_idx),
                epoch_id_hint=epoch_id,
                display_tag=label,
            )
        )
    return expanded


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


def _pick_trial_file(
    root: Path,
    *,
    prefix: str,
    cache_tag: str,
    trial_index: Optional[int],
) -> Optional[Path]:
    if trial_index is None:
        latest = _pick_latest(root, f"{prefix}_{cache_tag}_trial*.dat")
        if latest is not None:
            return latest
        return _pick_latest(root, f"{prefix}_{cache_tag}.dat")
    exact = root / f"{prefix}_{cache_tag}_trial{trial_index}.dat"
    if exact.exists():
        return exact
    return _pick_latest(root, f"{prefix}_{cache_tag}_trial{trial_index}*.dat")


def _traj_time_len(path: Path) -> int:
    try:
        idx = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(0,))
    except Exception:
        return 0
    arr = np.asarray(idx, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0
    return int(np.count_nonzero(np.isclose(arr, 0.0)))


def _trial_lengths_from_output(output_root: Path, cache_tag: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    try:
        paths = sorted(output_root.glob(f"traj_samples_{cache_tag}_trial*.dat"))
    except Exception:
        return out
    for path in paths:
        idx = _parse_trial_index_from_name(path.name)
        if idx is None:
            continue
        n = _traj_time_len(path)
        if n > 0:
            out[int(idx)] = int(n)
    return out


def _split_time_segments(time: np.ndarray) -> List[np.ndarray]:
    t = np.asarray(time, dtype=np.float64).ravel()
    if t.size <= 1:
        return [t]
    dt = np.diff(t)
    dt_pos = dt[np.isfinite(dt) & (dt > 0)]
    if dt_pos.size == 0:
        return [t]
    dt_ref = float(np.nanmedian(dt_pos))
    gap_threshold = max(5.0 * dt_ref, 1e-9)
    starts = [0]
    for idx, delta in enumerate(dt, start=1):
        if (not np.isfinite(delta)) or delta <= 0 or delta > gap_threshold:
            starts.append(idx)
    starts.append(t.size)
    out: List[np.ndarray] = []
    for start, end in zip(starts[:-1], starts[1:]):
        if end > start:
            out.append(t[start:end])
    return out if out else [t]


def _slice_run_time_for_trial(
    *,
    run_time: np.ndarray,
    output_root: Path,
    cache_tag: str,
    trial_index: int,
    expected_len: int,
) -> np.ndarray:
    t = np.asarray(run_time, dtype=np.float64).ravel()
    if t.size == 0 or trial_index < 0:
        return t

    lengths = _trial_lengths_from_output(output_root, cache_tag)
    if lengths:
        ordered = sorted(lengths.items())
        contiguous = [idx for idx, _n in ordered] == list(range(len(ordered)))
        if contiguous and trial_index < len(ordered):
            start = int(sum(length for _idx, length in ordered[:trial_index]))
            seg_len = int(ordered[trial_index][1])
            if start < t.size and seg_len > 0:
                end = min(start + seg_len, t.size)
                seg = t[start:end]
                if seg.size > 0:
                    return seg

    segments = _split_time_segments(t)
    if trial_index < len(segments):
        seg = segments[trial_index]
        if seg.size > 0:
            return seg

    if expected_len > 0:
        start = int(trial_index) * int(expected_len)
        if start < t.size:
            end = min(start + int(expected_len), t.size)
            seg = t[start:end]
            if seg.size > 0:
                return seg
    return t


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
        "time_len": time_len,
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
