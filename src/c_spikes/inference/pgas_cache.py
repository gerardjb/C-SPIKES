from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio


PGAS_SAMPLES_FIELD = "pgas_samples"


@dataclass(frozen=True)
class PgasTrajectorySamples:
    """Dense trajectory samples for one PGAS trial loaded from cache .mat."""

    trial_index: Optional[int]
    tag: str
    path: Optional[str]
    columns: Tuple[str, ...]
    n_samples: int
    n_time: int
    burnin: int
    time_stamps: np.ndarray
    values: Mapping[str, np.ndarray]
    post_burnin_mean: Mapping[str, np.ndarray]
    map: Mapping[str, Any]


@dataclass(frozen=True)
class PgasParameterSamples:
    """Parameter samples for one PGAS trial loaded from cache .mat."""

    trial_index: Optional[int]
    tag: str
    path: Optional[str]
    columns: Tuple[str, ...]
    burnin: int
    values: np.ndarray
    post_burnin_mean: np.ndarray


@dataclass(frozen=True)
class PgasLogpSamples:
    """Log-probability samples for one PGAS trial loaded from cache .mat."""

    trial_index: Optional[int]
    tag: str
    path: Optional[str]
    burnin: int
    values: np.ndarray
    post_burnin_mean: Optional[float]
    post_burnin_max: Optional[float]
    map_sample_index: Optional[int]


@dataclass(frozen=True)
class PgasSamplesCache:
    """Normalized PGAS sample payload embedded in a method cache .mat."""

    schema_version: int
    description: str
    mat_path: Optional[Path] = None
    trajectory_samples: Tuple[PgasTrajectorySamples, ...] = field(default_factory=tuple)
    parameter_samples: Tuple[PgasParameterSamples, ...] = field(default_factory=tuple)
    logp: Tuple[PgasLogpSamples, ...] = field(default_factory=tuple)

    def trajectory_for_trial(self, trial_index: int) -> Optional[PgasTrajectorySamples]:
        return _first_for_trial(self.trajectory_samples, trial_index)

    def parameters_for_trial(self, trial_index: int) -> Optional[PgasParameterSamples]:
        return _first_for_trial(self.parameter_samples, trial_index)

    def logp_for_trial(self, trial_index: int) -> Optional[PgasLogpSamples]:
        return _first_for_trial(self.logp, trial_index)


def has_pgas_samples(mat_path: Path | str) -> bool:
    """Return whether a cache .mat contains the embedded PGAS samples payload."""

    data = sio.loadmat(mat_path, variable_names=[PGAS_SAMPLES_FIELD])
    return PGAS_SAMPLES_FIELD in data


def load_pgas_samples_from_cache(
    mat_path: Path | str,
    *,
    require: bool = False,
) -> Optional[PgasSamplesCache]:
    """
    Load the embedded PGAS samples payload from a method cache .mat.

    Old caches do not contain ``pgas_samples``. For those, this returns ``None``
    unless ``require=True`` is supplied.
    """

    path = Path(mat_path)
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return pgas_samples_from_mat_data(data, mat_path=path, require=require)


def pgas_samples_from_mat_data(
    mat_data: Mapping[str, Any],
    *,
    mat_path: Optional[Path | str] = None,
    require: bool = False,
) -> Optional[PgasSamplesCache]:
    """Normalize a loaded cache ``.mat`` dictionary into PGAS sample dataclasses."""

    raw = mat_data.get(PGAS_SAMPLES_FIELD)
    if raw is None:
        if require:
            raise KeyError(f"Cache .mat does not contain '{PGAS_SAMPLES_FIELD}'.")
        return None
    payload = _mat_to_python(raw)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected '{PGAS_SAMPLES_FIELD}' to be a MATLAB struct.")
    return _build_pgas_samples(payload, mat_path=None if mat_path is None else Path(mat_path))


def _first_for_trial(items: Sequence[Any], trial_index: int) -> Optional[Any]:
    for item in items:
        if item.trial_index == int(trial_index):
            return item
    return None


def _build_pgas_samples(payload: Mapping[str, Any], *, mat_path: Optional[Path]) -> PgasSamplesCache:
    return PgasSamplesCache(
        schema_version=_optional_int(payload.get("schema_version")) or 0,
        description=_as_string(payload.get("description")) or "",
        mat_path=mat_path,
        trajectory_samples=tuple(
            _build_trajectory_sample(entry)
            for entry in _as_mapping_list(payload.get("trajectory_samples"))
        ),
        parameter_samples=tuple(
            _build_parameter_sample(entry)
            for entry in _as_mapping_list(payload.get("parameter_samples"))
        ),
        logp=tuple(_build_logp_sample(entry) for entry in _as_mapping_list(payload.get("logp"))),
    )


def _build_trajectory_sample(entry: Mapping[str, Any]) -> PgasTrajectorySamples:
    values: Dict[str, np.ndarray] = {}
    for field_name in ("index", "burst", "baseline", "spikes", "calcium", "observation"):
        if field_name in entry:
            values[field_name] = _numeric_array(entry.get(field_name), min_ndim=2)

    first_matrix = next(iter(values.values()), np.zeros((0, 0), dtype=np.float64))
    n_samples = _optional_int(entry.get("n_samples"))
    n_time = _optional_int(entry.get("n_time"))
    if n_samples is None and first_matrix.ndim >= 2:
        n_samples = int(first_matrix.shape[0])
    if n_time is None and first_matrix.ndim >= 2:
        n_time = int(first_matrix.shape[1])

    return PgasTrajectorySamples(
        trial_index=_optional_int(entry.get("trial_index")),
        tag=_as_string(entry.get("tag")) or "",
        path=_as_string(entry.get("path")),
        columns=_string_tuple(entry.get("columns")),
        n_samples=int(n_samples or 0),
        n_time=int(n_time or 0),
        burnin=int(_optional_int(entry.get("burnin")) or 0),
        time_stamps=_numeric_array(entry.get("time_stamps"), min_ndim=1).ravel(),
        values=values,
        post_burnin_mean=_numeric_mapping(entry.get("post_burnin_mean")),
        map=_mixed_numeric_mapping(entry.get("map")),
    )


def _build_parameter_sample(entry: Mapping[str, Any]) -> PgasParameterSamples:
    return PgasParameterSamples(
        trial_index=_optional_int(entry.get("trial_index")),
        tag=_as_string(entry.get("tag")) or "",
        path=_as_string(entry.get("path")),
        columns=_string_tuple(entry.get("columns")),
        burnin=int(_optional_int(entry.get("burnin")) or 0),
        values=_numeric_array(entry.get("values"), min_ndim=2),
        post_burnin_mean=_numeric_array(entry.get("post_burnin_mean"), min_ndim=1).ravel(),
    )


def _build_logp_sample(entry: Mapping[str, Any]) -> PgasLogpSamples:
    return PgasLogpSamples(
        trial_index=_optional_int(entry.get("trial_index")),
        tag=_as_string(entry.get("tag")) or "",
        path=_as_string(entry.get("path")),
        burnin=int(_optional_int(entry.get("burnin")) or 0),
        values=_numeric_array(entry.get("values"), min_ndim=1).ravel(),
        post_burnin_mean=_optional_float(entry.get("post_burnin_mean")),
        post_burnin_max=_optional_float(entry.get("post_burnin_max")),
        map_sample_index=_optional_int(entry.get("map_sample_index")),
    )


def _mat_to_python(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return {
            field_name: _mat_to_python(getattr(value, field_name))
            for field_name in value._fieldnames
        }
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.shape == ():
                return _mat_to_python(value.item())
            return [_mat_to_python(item) for item in value.ravel()]
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_mapping_list(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, Mapping))
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _as_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.dtype.kind in {"U", "S"}:
        if arr.ndim == 0:
            item = arr.item()
            return item.decode("utf-8") if isinstance(item, bytes) else str(item)
        return "".join(str(item) for item in arr.ravel())
    if arr.ndim == 0:
        return str(arr.item())
    return str(value)


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        text = _as_string(value)
        return () if text is None else (text.strip(),)
    if isinstance(value, list):
        return tuple(label for item in value if (label := str(item).strip()))
    arr = np.asarray(value)
    if arr.size == 0:
        return ()
    return tuple(label for item in arr.ravel() if (label := str(item).strip()))


def _numeric_array(value: Any, *, min_ndim: int) -> np.ndarray:
    if value is None:
        arr = np.asarray([], dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if min_ndim <= 1:
        return np.atleast_1d(arr)
    if arr.ndim == 0:
        return arr.reshape((1, 1))
    if arr.ndim == 1:
        return arr.reshape((1, -1))
    return arr


def _numeric_mapping(value: Any) -> Dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, np.ndarray] = {}
    for key, item in value.items():
        out[str(key)] = _numeric_array(item, min_ndim=1).ravel()
    return out


def _mixed_numeric_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key, item in value.items():
        arr = _numeric_array(item, min_ndim=1)
        if arr.size == 1:
            out[str(key)] = float(arr.ravel()[0])
        else:
            out[str(key)] = arr.ravel()
    return out


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return int(arr.ravel()[0])
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return float(arr.ravel()[0])
    except (TypeError, ValueError):
        return None


__all__ = [
    "PGAS_SAMPLES_FIELD",
    "PgasLogpSamples",
    "PgasParameterSamples",
    "PgasSamplesCache",
    "PgasTrajectorySamples",
    "has_pgas_samples",
    "load_pgas_samples_from_cache",
    "pgas_samples_from_mat_data",
]
