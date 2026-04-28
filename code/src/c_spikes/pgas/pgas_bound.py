"""Runtime shim that selects the CPU or GPU PGAS backend."""

from __future__ import annotations

import importlib
import os
from typing import Dict, Optional

_BACKEND_ENV = os.environ.get("C_SPIKES_PGAS_BACKEND", "").strip().lower()
_BACKEND_ALIASES = {
    "gpu": "gpu",
    "cuda": "gpu",
    "kokkos": "gpu",
    "cpu": "cpu",
    "host": "cpu",
}
_BACKEND_ORDER = ("gpu", "cpu")


def _normalize_backend(choice: str) -> str:
    normalized = _BACKEND_ALIASES.get(choice)
    if normalized is None:
        raise ValueError(
            "Unknown C_SPIKES_PGAS_BACKEND value: "
            f"'{choice}'. Use 'cpu' or 'gpu'."
        )
    return normalized


def _import_backend(choice: str):
    module_name = f"c_spikes.pgas.pgas_bound_{choice}"
    return importlib.import_module(module_name)


_backend = None
_backend_name: Optional[str] = None
_errors: Dict[str, Exception] = {}

if _BACKEND_ENV:
    _backend_name = _normalize_backend(_BACKEND_ENV)
    try:
        _backend = _import_backend(_backend_name)
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise ImportError(
            f"Failed to import PGAS backend '{_backend_name}'."
        ) from exc
else:
    for candidate in _BACKEND_ORDER:
        try:
            _backend = _import_backend(candidate)
            _backend_name = candidate
            break
        except (ModuleNotFoundError, ImportError, OSError) as exc:
            _errors[candidate] = exc
    if _backend is None:
        msg = "Unable to import any PGAS backend."
        if _errors:
            details = "; ".join(f"{k}: {type(v).__name__}" for k, v in _errors.items())
            msg = f"{msg} Tried: {details}."
        raise ImportError(msg)

__backend__ = _backend_name

__all__ = []
for _name in dir(_backend):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_backend, _name)
    __all__.append(_name)
