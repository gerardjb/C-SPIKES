import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

from c_spikes.inference.cascade import CascadeConfig, run_cascade_inference
from c_spikes.inference.ens2 import Ens2Config, run_ens2_inference
from c_spikes.inference.types import TrialSeries

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.requires_tensorflow
def test_cascade_inference_smoke(tmp_path, monkeypatch):
    repo_root = _repo_root()
    model_dir = repo_root / "Pretrained_models"
    model_path = model_dir / "Cascade_Universal_30Hz"
    assert model_path.exists(), f"Missing CASCADE model folder: {model_path}"

    rng = np.random.default_rng(0)
    n_samples = 128
    fs = 30.0
    times = np.arange(n_samples, dtype=np.float64) / fs
    values = rng.normal(0.0, 0.05, size=n_samples).astype(np.float64)

    trials = [TrialSeries(times=times, values=values)]
    config = CascadeConfig(
        dataset_tag="pytest_synth",
        model_folder=model_dir,
        model_name="Cascade_Universal_30Hz",
        resample_fs=fs,
        downsample_label="raw",
        use_cache=False,
        discretize=False,
    )

    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        result = run_cascade_inference(trials, config)

    assert result.name == "cascade"
    assert result.time_stamps.size == result.spike_prob.size
    assert np.isfinite(result.sampling_rate)
    assert np.isfinite(result.spike_prob).any()
    logger.info("CASCADE inference produced %d samples", result.spike_prob.size)


@pytest.mark.slow
@pytest.mark.requires_torch
def test_ens2_inference_smoke(tmp_path, monkeypatch):
    repo_root = _repo_root()
    pretrained_dir = repo_root / "Pretrained_models" / "ens2_published"
    assert pretrained_dir.exists(), f"Missing ENS2 pretrained dir: {pretrained_dir}"

    rng = np.random.default_rng(1)
    n_samples = 120
    fs = 30.0
    times = np.arange(n_samples, dtype=np.float64) / fs
    raw_time_stamps = np.stack([times], axis=0)
    raw_traces = rng.normal(0.0, 0.05, size=(1, n_samples)).astype(np.float64)

    config = Ens2Config(
        dataset_tag="pytest_synth",
        pretrained_dir=pretrained_dir,
        neuron_type="Exc",
        downsample_label="raw",
        use_cache=False,
    )

    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import torch

        torch.set_num_threads(1)
        result = run_ens2_inference(raw_time_stamps, raw_traces, config)

    assert result.name == "ens2"
    assert result.time_stamps.size == result.spike_prob.size
    assert np.isfinite(result.sampling_rate)
    assert np.isfinite(result.spike_prob).any()
    logger.info("ENS2 inference produced %d samples", result.spike_prob.size)
