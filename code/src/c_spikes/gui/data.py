from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import threading

import numpy as np
import scipy.io as sio

try:  # Optional for MATLAB v7.3 files
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore


@dataclass(frozen=True)
class MatFileInfo:
    path: Path
    n_epochs: int
    has_spikes: bool
    dff_shape: Tuple[int, ...]
    time_shape: Tuple[int, ...]


@dataclass(frozen=True)
class EpochRef:
    file_path: Path
    epoch_index: int
    epoch_count: int
    epoch_id: str
    display: str
    has_spikes: Optional[bool] = None


@dataclass
class LoadedMat:
    path: Path
    time_stamps: np.ndarray
    dff: np.ndarray
    ap_times: Optional[np.ndarray]


class DataManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Optional[LoadedMat] = None

    def load_epoch(self, epoch: EpochRef) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        with self._lock:
            if self._cache is None or self._cache.path != epoch.file_path:
                self._cache = _load_mat_data(epoch.file_path)
            loaded = self._cache
        return extract_epoch_data(loaded, epoch)


def scan_dataset_dir(data_dir: Path) -> Tuple[List[EpochRef], List[str]]:
    data_dir = Path(data_dir)
    errors: List[str] = []
    epochs: List[EpochRef] = []
    if not data_dir.exists():
        return [], [f"Data directory does not exist: {data_dir}"]

    mat_paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".mat")
    for mat_path in mat_paths:
        try:
            info = scan_mat_file(mat_path)
        except Exception as exc:
            errors.append(f"{mat_path.name}: {exc}")
            continue

        for idx in range(info.n_epochs):
            epoch_id = f"{mat_path.stem}_epoch{idx + 1}"
            display = f"{mat_path.stem} [epoch {idx + 1}/{info.n_epochs}]"
            epochs.append(
                EpochRef(
                    file_path=mat_path,
                    epoch_index=idx,
                    epoch_count=info.n_epochs,
                    epoch_id=epoch_id,
                    display=display,
                    has_spikes=info.has_spikes,
                )
            )
    return epochs, errors


def scan_mat_file(path: Path) -> MatFileInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        vars_info = sio.whosmat(path)
        names = {name: shape for name, shape, _ in vars_info}
        if "dff" not in names:
            raise KeyError("Missing 'dff' array")
        dff_shape = tuple(int(v) for v in names["dff"])
        time_shape = tuple(int(v) for v in names.get("time_stamps", ()))
        n_epochs = _infer_epoch_count(dff_shape)
        has_spikes = "ap_times" in names
        return MatFileInfo(
            path=path,
            n_epochs=n_epochs,
            has_spikes=has_spikes,
            dff_shape=dff_shape,
            time_shape=time_shape,
        )
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError("MATLAB v7.3 file requires h5py, which is unavailable.")
        with h5py.File(path, "r") as h5:
            if "dff" not in h5:
                raise KeyError("Missing 'dff' dataset")
            dff_shape = tuple(int(v) for v in h5["dff"].shape)
            time_shape = tuple(int(v) for v in h5["time_stamps"].shape) if "time_stamps" in h5 else ()
            n_epochs = _infer_epoch_count(dff_shape)
            has_spikes = "ap_times" in h5
            return MatFileInfo(
                path=path,
                n_epochs=n_epochs,
                has_spikes=has_spikes,
                dff_shape=dff_shape,
                time_shape=time_shape,
            )


def _infer_epoch_count(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 0
    if len(shape) == 1:
        return 1
    if len(shape) == 2:
        return int(min(shape))
    return int(shape[0])


def _load_mat_data(path: Path) -> LoadedMat:
    path = Path(path)
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        dff = data.get("dff")
        time_stamps = data.get("time_stamps")
        ap_times = data.get("ap_times")
        if dff is None or time_stamps is None:
            raise KeyError("Missing required keys 'dff' and/or 'time_stamps'")
        return LoadedMat(
            path=path,
            time_stamps=np.asarray(time_stamps),
            dff=np.asarray(dff),
            ap_times=None if ap_times is None else np.asarray(ap_times),
        )
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError("MATLAB v7.3 file requires h5py, which is unavailable.")
        with h5py.File(path, "r") as h5:
            if "dff" not in h5 or "time_stamps" not in h5:
                raise KeyError("Missing required keys 'dff' and/or 'time_stamps'")
            dff = np.array(h5["dff"])
            time_stamps = np.array(h5["time_stamps"])
            ap_times = None
            if "ap_times" in h5:
                try:
                    ap_times = np.array(h5["ap_times"])
                except Exception:
                    ap_times = None
            # MATLAB v7.3 arrays are column-major; transpose common 2D arrays.
            if dff.ndim == 2:
                dff = dff.T
            if time_stamps.ndim == 2:
                time_stamps = time_stamps.T
            return LoadedMat(
                path=path,
                time_stamps=np.asarray(time_stamps),
                dff=np.asarray(dff),
                ap_times=None if ap_times is None else np.asarray(ap_times, dtype=object),
            )


def extract_epoch_data(
    loaded: LoadedMat,
    epoch: EpochRef,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    n_epochs = epoch.epoch_count
    time = _extract_epoch_array(loaded.time_stamps, epoch.epoch_index, n_epochs)
    dff = _extract_epoch_array(loaded.dff, epoch.epoch_index, n_epochs)
    spikes = _extract_spike_times(loaded.ap_times, epoch.epoch_index, n_epochs)

    time = np.asarray(time, dtype=np.float64).ravel()
    dff = np.asarray(dff, dtype=np.float64).ravel()
    mask = np.isfinite(time) & np.isfinite(dff)
    time = time[mask]
    dff = dff[mask]
    if time.size == 0:
        raise ValueError("Epoch contains no finite time stamps")
    return time, dff, spikes


def _extract_epoch_array(arr: np.ndarray, epoch_idx: int, n_epochs: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == n_epochs:
            return arr[epoch_idx]
        if arr.shape[1] == n_epochs:
            return arr[:, epoch_idx]
        return arr[epoch_idx]
    return arr[epoch_idx]


def _extract_spike_times(
    ap_times: Optional[np.ndarray],
    epoch_idx: int,
    n_epochs: int,
) -> Optional[np.ndarray]:
    if ap_times is None:
        return None
    arr = np.asarray(ap_times)
    if arr.size == 0:
        return None
    if arr.dtype == object:
        flat = arr.ravel()
        if flat.size == n_epochs:
            return np.asarray(flat[epoch_idx]).ravel()
        if flat.size == 1:
            return np.asarray(flat[0]).ravel()
        try:
            numeric = np.asarray(flat, dtype=np.float64)
            return numeric.ravel()
        except Exception:
            parts = []
            for item in flat:
                if item is None:
                    continue
                try:
                    parts.append(np.asarray(item, dtype=np.float64).ravel())
                except Exception:
                    continue
            if parts:
                return np.concatenate(parts)
            return None
    if arr.ndim == 1:
        return arr.ravel()
    if arr.ndim == 2 and arr.shape[0] == n_epochs:
        return np.asarray(arr[epoch_idx]).ravel()
    if arr.ndim == 2 and arr.shape[1] == n_epochs:
        return np.asarray(arr[:, epoch_idx]).ravel()
    return arr.ravel()
