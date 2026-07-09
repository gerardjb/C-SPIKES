# TEMPORARY PORT PARITY TESTS:
# Remove after main_cache_noise_port is validated and the port commits are accepted.

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.io as sio

from c_spikes.inference.noise_calibration import (
    detect_narrowband_psd_peaks,
    estimate_observation_noise_psd,
    run_psd_notch_preprocess_mat,
)
from c_spikes.inference.pgas import estimate_noise_calibration_for_trials
from c_spikes.inference.types import TrialSeries


pytestmark = [
    pytest.mark.port_parity,
    pytest.mark.skipif(
        os.environ.get("C_SPIKES_RUN_PORT_PARITY") != "1",
        reason="Temporary port parity test; set C_SPIKES_RUN_PORT_PARITY=1 to run.",
    ),
]


def test_psd_white_noise_variance_estimate_matches_known_scale():
    rng = np.random.default_rng(42)
    fs_hz = 100.0
    expected_var = 0.004
    y = rng.normal(scale=np.sqrt(expected_var), size=20000)

    estimate = estimate_observation_noise_psd(
        y,
        fs_hz=fs_hz,
        nperseg=2048,
        min_frequency_hz=1.0,
    )

    assert estimate.variance == pytest.approx(expected_var, rel=0.25)
    assert estimate.n_frequency_bins > 10


def test_psd_detects_and_excludes_sinusoidal_peak():
    rng = np.random.default_rng(123)
    fs_hz = 100.0
    expected_var = 0.003
    t = np.arange(20000, dtype=float) / fs_hz
    y = rng.normal(scale=np.sqrt(expected_var), size=t.size) + 0.5 * np.sin(2.0 * np.pi * 6.0 * t)

    _freq, _psd_db, targets = detect_narrowband_psd_peaks(
        y,
        fs_hz=fs_hz,
        nperseg=4096,
        prominence_db=10.0,
        max_peak_width_hz=1.0,
        min_frequency_hz=1.0,
        max_frequency_hz=20.0,
    )
    estimate = estimate_observation_noise_psd(
        y,
        fs_hz=fs_hz,
        nperseg=4096,
        min_frequency_hz=1.0,
        max_frequency_hz=40.0,
        peak_prominence_db=10.0,
        max_peak_width_hz=1.0,
    )

    assert any(abs(target.frequency_hz - 6.0) < 0.1 for target in targets)
    assert any(abs(target.frequency_hz - 6.0) < 0.1 for target in estimate.peaks)
    assert estimate.n_excluded_bins > 0
    assert estimate.variance == pytest.approx(expected_var, rel=0.35)


def test_psd_mode_tracks_observation_scale_under_random_walk_drift():
    rng = np.random.default_rng(7)
    fs_hz = 100.0
    expected_var = 0.0025
    t = np.arange(20000, dtype=float) / fs_hz
    drift = np.cumsum(rng.normal(scale=0.01, size=t.size))
    y = drift + rng.normal(scale=np.sqrt(expected_var), size=t.size)
    trial = TrialSeries(times=t, values=y)

    diff_cal = estimate_noise_calibration_for_trials(
        [trial],
        spike_times=np.array([], dtype=float),
        resample_fs=fs_hz,
        gap_s=0.15,
        method="diff",
    )
    psd_cal = estimate_noise_calibration_for_trials(
        [trial],
        spike_times=np.array([], dtype=float),
        resample_fs=fs_hz,
        gap_s=0.15,
        method="psd",
    )

    assert diff_cal.diff_var > expected_var
    assert psd_cal.method == "psd"
    assert psd_cal.sigma2_target == pytest.approx(expected_var, rel=0.5)


def test_psd_notch_script_wrapper_writes_mat_and_diagnostics(tmp_path):
    fs_hz = 100.0
    t = np.arange(1000, dtype=float) / fs_hz
    y = 0.1 * np.sin(2.0 * np.pi * 6.0 * t)
    input_mat = tmp_path / "synthetic_cell.mat"
    output_mat = tmp_path / "synthetic_cell_psdnotch_e01.mat"
    plot_dir = tmp_path / "plots"
    sio.savemat(
        input_mat,
        {
            "time_stamps": t.reshape(1, -1),
            "dff": y.reshape(1, -1),
            "ap_times": np.array([[0.5, 1.5]], dtype=float),
        },
    )

    rc = run_psd_notch_preprocess_mat(
        [
            "--input-mat",
            str(input_mat),
            "--output-mat",
            str(output_mat),
            "--plot-dir",
            str(plot_dir),
            "--epoch",
            "1",
            "--welch-nperseg",
            "512",
            "--peak-prominence-db",
            "10",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert output_mat.exists()
    payload = sio.loadmat(output_mat)
    assert payload["dff"].shape == (1, t.size)
    assert payload["psd_notch_frequencies_hz"].size >= 1
    assert list(plot_dir.glob("*_psd.png"))
    assert list(plot_dir.glob("*_timeseries.png"))
    assert list(plot_dir.glob("*_peaks.csv"))
