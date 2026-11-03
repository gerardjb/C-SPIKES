"""Load an independent PGAS backend when it is requested."""
from __future__ import annotations

import importlib
import os


def backend_name(backend: str | None = None) -> str:
    choice = backend if backend is not None else os.environ.get("C_SPIKES_PGAS_BACKEND", "cpu")
    choice = choice.strip().lower() or "cpu"
    aliases = {"cpu": "cpu", "host": "cpu", "gpu": "gpu", "cuda": "gpu", "kokkos": "gpu"}
    if choice not in aliases:
        raise ValueError(f"Unknown PGAS backend {choice!r}; use 'cpu' or 'cuda'")
    return aliases[choice]


def get_backend(backend: str | None = None):
    """Return the CPU or CUDA native module; CPU is the default.

    Modules are cached by Python. Both can coexist; selecting another backend
    does not move or invalidate objects created by the previous backend.
    """
    name = backend_name(backend)
    try:
        return importlib.import_module(f"c_spikes.pgas.pgas_bound_{name}")
    except (ImportError, OSError) as exc:
        if name == "gpu":
            raise ImportError(
                "CUDA backend unavailable. Install a CUDA-enabled c-spikes wheel and a "
                "compatible NVIDIA driver, and make a GPU visible; use get_backend('cpu') "
                "for CPU execution."
            ) from exc
        raise
