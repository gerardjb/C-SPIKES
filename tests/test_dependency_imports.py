import logging
import os
import warnings

import pytest

logger = logging.getLogger(__name__)


def _import_torch():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import torch  # type: ignore
    return torch


def _import_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import tensorflow as tf  # type: ignore
    return tf


@pytest.mark.requires_torch
def test_torch_import_and_ens2_module():
    try:
        torch = _import_torch()
    except Exception as exc:
        pytest.fail(f"Torch import failed: {exc}")

    try:
        ens2_module = __import__("c_spikes.ens2.ENS2", fromlist=["ENS2"])
    except Exception as exc:
        pytest.fail(f"ENS2 module import failed: {exc}")

    assert hasattr(ens2_module, "ENS2")
    cuda_available = bool(torch.cuda.is_available())
    logger.info("Torch version=%s CUDA available=%s", torch.__version__, cuda_available)
    if cuda_available and torch.version.cuda is not None:
        logger.info("Torch CUDA version=%s", torch.version.cuda)


@pytest.mark.requires_tensorflow
def test_tensorflow_import_and_cascade_module():
    try:
        tf = _import_tensorflow()
    except Exception as exc:
        pytest.fail(f"TensorFlow import failed: {exc}")

    try:
        cascade_module = __import__("c_spikes.cascade2p.cascade", fromlist=["predict"])
    except Exception as exc:
        pytest.fail(f"CASCADE module import failed: {exc}")

    assert hasattr(cascade_module, "predict")
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [gpu.name for gpu in gpus]
    logger.info("TensorFlow version=%s GPUs=%s", tf.__version__, gpu_names)
