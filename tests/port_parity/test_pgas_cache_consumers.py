# TEMPORARY PORT PARITY TESTS:
# Remove after main_cache_noise_port is validated and the port commits are accepted.

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.io as sio

from c_spikes.inference.pgas import build_pgas_output_cache_payload, load_pgas_method_result
from c_spikes.inference.pgas_cache import load_pgas_samples_from_cache
from c_spikes.inference.types import TrialSeries
from c_spikes.syn_gen import _load_pgas_parameter_samples
from c_spikes.viz.trialwise_plots import DEFAULT_COLORS, DEFAULT_LABELS


pytestmark = [
    pytest.mark.port_parity,
    pytest.mark.skipif(
        os.environ.get("C_SPIKES_RUN_PORT_PARITY") != "1",
        reason="Temporary port parity test; set C_SPIKES_RUN_PORT_PARITY=1 to run.",
    ),
]


def _write_synthetic_pgas_outputs(root, tag: str) -> None:
    with (root / f"traj_samples_{tag}.dat").open("w", encoding="utf-8") as fh:
        fh.write("index,burst,B,S,C,Y\n")
        for sample in range(3):
            for t in range(4):
                fh.write(
                    f"{sample},{sample % 2},{sample + 0.1 * t:.6f},{sample + t:.6f},"
                    f"{2 * sample + t:.6f},{10 + t:.6f}\n"
                )
    with (root / f"param_samples_{tag}.dat").open("w", encoding="utf-8") as fh:
        fh.write("r0,r1,sigma2\n")
        for sample in range(3):
            fh.write(f"{sample + 1:.6f},{sample + 2:.6f},{0.1 * (sample + 1):.6f}\n")
    np.savetxt(root / f"logp_{tag}.dat", np.asarray([-3.0, -1.0, -2.0]))


def test_pgas_method_result_and_syn_gen_read_from_embedded_cache(tmp_path):
    dataset_tag = "cache_consumer"
    trial_tag = f"{dataset_tag}_trial0"
    _write_synthetic_pgas_outputs(tmp_path, trial_tag)
    trial = TrialSeries(times=np.asarray([0.0, 1.0, 2.0, 3.0]), values=np.zeros(4))
    payload = build_pgas_output_cache_payload([trial], dataset_tag, tmp_path, burnin=1)
    mat_path = tmp_path / "cache.mat"
    sio.savemat(mat_path, payload, do_compression=True)

    samples = load_pgas_samples_from_cache(mat_path, require=True)
    result = load_pgas_method_result(
        [trial],
        dataset_tag,
        tmp_path / "missing_dat_dir",
        burnin=1,
        cache_mat_path=mat_path,
        samples_cache=samples,
    )

    np.testing.assert_allclose(result.spike_prob, np.asarray([1.5, 2.5, 3.5, 4.5]))
    np.testing.assert_allclose(result.discrete_spikes, np.asarray([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(result.reconstruction, np.asarray([4.5, 5.6, 6.7, 7.8]))

    params, tag = _load_pgas_parameter_samples(mat_path, trial_index=0)
    assert tag == trial_tag
    assert params.shape == (3, 3)
    np.testing.assert_allclose(params[1], np.asarray([2.0, 3.0, 0.2]))


def test_trialwise_plot_defaults_include_pgbar():
    assert DEFAULT_COLORS["pgbar"] == "#4A4A4A"
    assert DEFAULT_LABELS["pgbar"] == "PGBAR"
