# TEMPORARY PORT PARITY TESTS:
# Remove after main_cache_noise_port is validated and the port commits are accepted.

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from c_spikes.inference.cache import set_cache_root
from c_spikes.inference.pgas import (
    PGAS_BM_SIGMA_DEFAULT,
    PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT,
    PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT,
    PgasConfig,
    _build_pgas_noise_settings,
    derive_bm_sigma_and_sigma2,
    map_sigma2_target_to_ig_params,
    normalize_noise_calibration_granularity,
    normalize_noise_calibration_method,
    normalize_noise_calibration_scope,
)
from c_spikes.inference.types import TrialSeries


pytestmark = [
    pytest.mark.port_parity,
    pytest.mark.skipif(
        os.environ.get("C_SPIKES_RUN_PORT_PARITY") != "1",
        reason="Temporary port parity test; set C_SPIKES_RUN_PORT_PARITY=1 to run.",
    ),
]


def _constants_file(tmp_path):
    path = tmp_path / "constants.json"
    path.write_text(
        json.dumps(
            {
                "MCMC": {"maxspikes": 2},
                "BM": {"bm_sigma": 0.02},
                "priors": {"alpha sigma2": 2.0, "beta sigma2": 0.01},
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(tmp_path, **kwargs):
    defaults = dict(
        dataset_tag="unit",
        output_root=tmp_path / "pgas_output",
        constants_file=_constants_file(tmp_path),
        gparam_file=tmp_path / "gparam.dat",
        bm_sigma=PGAS_BM_SIGMA_DEFAULT,
    )
    defaults.update(kwargs)
    return PgasConfig(**defaults)


def test_fixed_noise_settings_preserve_main_defaults(tmp_path):
    set_cache_root(tmp_path / "cache")
    trial = TrialSeries(
        times=np.linspace(0.0, 1.0, 11),
        values=np.linspace(0.0, 0.1, 11),
    )
    settings = _build_pgas_noise_settings(
        config=_config(tmp_path),
        maxspikes=2,
        calibration_trials=[trial],
        spike_times=np.array([], dtype=float),
        input_fs=10.0,
        bm_sigma_min=5e-4,
        bm_sigma_max=0.5,
    )

    assert settings.bm_sigma == pytest.approx(PGAS_BM_SIGMA_DEFAULT)
    assert settings.sigma2_target is None
    assert settings.calibration is None
    assert "noise_calibration" not in settings.cfg
    assert settings.cfg["bm_sigma"] == pytest.approx(PGAS_BM_SIGMA_DEFAULT)


def test_auto_noise_settings_record_bm_clipping(tmp_path):
    set_cache_root(tmp_path / "cache")
    rng = np.random.default_rng(123)
    times = np.linspace(0.0, 2.0, 201)
    values = np.cumsum(rng.normal(scale=0.2, size=times.size))
    trial = TrialSeries(times=times, values=values)

    settings = _build_pgas_noise_settings(
        config=_config(tmp_path, bm_sigma=None, bm_sigma_min=1e-4, bm_sigma_max=1e-3),
        maxspikes=2,
        calibration_trials=[trial],
        spike_times=np.array([], dtype=float),
        input_fs=100.0,
        bm_sigma_min=1e-4,
        bm_sigma_max=1e-3,
    )

    assert settings.calibration is not None
    assert settings.bm_sigma == pytest.approx(1e-3)
    assert settings.noise_calibration is not None
    assert settings.noise_calibration["clipped_bm_sigma"] is True
    assert settings.noise_calibration["bm_sigma_unclipped"] > settings.bm_sigma
    assert settings.cfg["noise_calibration"]["clipped_bm_sigma"] is True


def test_sigma2_target_mapping_and_calibration_normalizers():
    target, alpha, beta = map_sigma2_target_to_ig_params(
        0.0025,
        sigma2_alpha=None,
        sigma2_prior_strength=10.0,
    )
    assert target == pytest.approx(0.0025)
    assert alpha == pytest.approx(12.0)
    assert beta == pytest.approx(0.0025 * 13.0)

    assert normalize_noise_calibration_scope("FULL") == "full"
    assert normalize_noise_calibration_scope(None or PGAS_NOISE_CALIBRATION_SCOPE_DEFAULT) == "inference"
    assert normalize_noise_calibration_granularity("TRIAL") == "trial"
    assert normalize_noise_calibration_method("PSD") == "psd"
    assert (
        normalize_noise_calibration_granularity(PGAS_NOISE_CALIBRATION_GRANULARITY_DEFAULT)
        == "dataset"
    )

    with pytest.raises(ValueError):
        normalize_noise_calibration_scope("edges")
    with pytest.raises(ValueError):
        normalize_noise_calibration_granularity("epoch")
    with pytest.raises(ValueError):
        normalize_noise_calibration_method("allan")


def test_derive_bm_sigma_and_sigma2_reports_sigma2_clipping():
    times = np.linspace(0.0, 1.0, 101)
    values = np.zeros_like(times)
    calibration = derive_bm_sigma_and_sigma2(
        times,
        values,
        target_fs=100.0,
        min_bm_sigma=1e-4,
        max_bm_sigma=0.5,
        min_sigma2_target=1e-5,
        max_sigma2_target=1e-2,
    )
    assert calibration.sigma2_target == pytest.approx(1e-5)
    assert calibration.sigma2_target_unclipped == pytest.approx(0.0)
    assert calibration.clipped_sigma2_target is True
