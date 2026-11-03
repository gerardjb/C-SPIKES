"""PGAS inference with lazily loaded CPU and CUDA backends."""
from .backends import get_backend

__all__ = ["get_backend"]
