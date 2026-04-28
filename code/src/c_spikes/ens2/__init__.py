from __future__ import annotations

from typing import TYPE_CHECKING

from .manifest import add_synthetic_entry, add_training_entry

if TYPE_CHECKING:  # pragma: no cover
    from .ens import predict as predict
    from .ens import train_model as train_model

__all__ = ["predict", "train_model", "add_synthetic_entry", "add_training_entry"]


def train_model(*args, **kwargs):
    """
    Lazily import the torch-dependent training wrapper.

    This keeps `c_spikes.ens2.manifest` and syn_gen workflows importable in
    environments where `torch` is not installed.
    """

    from .ens import train_model as _train_model

    return _train_model(*args, **kwargs)


def predict(*args, **kwargs):
    """
    Lazily import the torch-dependent inference wrapper.

    This keeps `c_spikes.ens2.manifest` and syn_gen workflows importable in
    environments where `torch` is not installed.
    """

    from .ens import predict as _predict

    return _predict(*args, **kwargs)
