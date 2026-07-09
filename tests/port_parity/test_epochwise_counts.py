# TEMPORARY PORT PARITY TESTS:
# Remove after main_cache_noise_port is validated and the port commits are accepted.

from __future__ import annotations

import os

import numpy as np
import pytest

from c_spikes.inference.eval import compute_epochwise_counts
from c_spikes.inference.types import MethodResult


pytestmark = [
    pytest.mark.port_parity,
    pytest.mark.skipif(
        os.environ.get("C_SPIKES_RUN_PORT_PARITY") != "1",
        reason="Temporary port parity test; set C_SPIKES_RUN_PORT_PARITY=1 to run.",
    ),
]


def test_epochwise_counts_follow_requested_windows_and_method_samples():
    pgas = MethodResult(
        name="pgas",
        time_stamps=np.asarray([0.0, 0.5, 1.0, 2.0, 2.5, 3.0]),
        spike_prob=np.zeros(6),
        sampling_rate=2.0,
        discrete_spikes=np.asarray([1.0, np.nan, 2.0, 0.0, 3.0, 1.0]),
    )
    no_discrete = MethodResult(
        name="ens2",
        time_stamps=np.asarray([0.0, 1.0, 2.0]),
        spike_prob=np.zeros(3),
        sampling_rate=1.0,
        discrete_spikes=None,
    )
    spike_times = np.asarray([0.1, 0.9, 1.8, 2.2, 3.2])

    full = compute_epochwise_counts(
        [pgas, no_discrete],
        spike_times,
        [(0.0, 1.0), (2.0, 3.0)],
    )
    assert full == {
        "gt_count": [2, 1],
        "pgas_samples": [3, 4],
        "ens2_samples": [0, 0],
    }

    edged = compute_epochwise_counts([pgas], spike_times, [(0.2, 0.8), (2.1, 2.6)])
    assert edged == {
        "gt_count": [0, 1],
        "pgas_samples": [0, 3],
    }
