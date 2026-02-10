"""Helpers to configure TensorFlow runtime logging/behavior before import."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from contextlib import contextmanager


def configure_tensorflow_environment() -> None:
    """Set conservative default TensorFlow env vars if user has not set them."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
    # Suppress TensorFlow Python logger warnings (e.g. retracing chatter).
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


@contextmanager
def _suppress_stderr_fd():
    """Temporarily redirect process-level stderr to /dev/null."""
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(null_fd)


def preload_tensorflow_quietly() -> None:
    """Import TensorFlow once while temporarily silencing early C++ startup logs."""
    configure_tensorflow_environment()

    if "tensorflow" in sys.modules:
        return

    if os.environ.get("C_SPIKES_TF_QUIET_IMPORT", "1") != "1":
        return

    try:
        with _suppress_stderr_fd():
            importlib.import_module("tensorflow")
    except Exception:
        # Retry without suppression so diagnostics are visible on real import failures.
        importlib.import_module("tensorflow")


@contextmanager
def suppress_tensorflow_stderr_if_configured():
    """Optionally suppress TensorFlow C++ stderr chatter during runtime calls."""
    if os.environ.get("C_SPIKES_TF_SUPPRESS_RUNTIME_STDERR", "1") != "1":
        yield
        return

    with _suppress_stderr_fd():
        yield
