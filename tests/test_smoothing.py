import numpy as np

from c_spikes.inference.smoothing import mean_downsample_trace, resample_trial_to_fs
from c_spikes.inference.types import TrialSeries


def test_noninteger_downsampling_uses_an_exact_target_rate_for_unequal_trials():
    outputs = []
    for length in (101, 106):
        times = np.arange(length, dtype=np.float64) / 100.0
        values = np.sin(times)
        outputs.append(mean_downsample_trace(times, values, 30.0))

    for output in outputs:
        np.testing.assert_allclose(np.diff(output.times), 1.0 / 30.0, rtol=0, atol=1e-15)
    assert outputs[0].times.size == 31
    assert outputs[1].times.size == 32
    assert outputs[1].times[-1] < 1.05


def test_upsampling_uses_an_exact_target_rate_without_stretching_to_endpoint():
    times = np.arange(12, dtype=np.float64) / 10.0
    trial = TrialSeries(times=times, values=np.cos(times))

    output = resample_trial_to_fs(trial, 25.0)

    np.testing.assert_allclose(np.diff(output.times), 1.0 / 25.0, rtol=0, atol=1e-15)
    assert output.times[-1] < times[-1]
    assert output.times[-1] + 1.0 / 25.0 > times[-1]
