"""Lazy compatibility shim. Use get_backend() for explicit per-object selection."""
from .backends import backend_name, get_backend


def __getattr__(name):
    if name == "__backend__":
        return backend_name()
    if name == "__all__":
        return [key for key in dir(get_backend()) if not key.startswith("_")]
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(get_backend(), name)
