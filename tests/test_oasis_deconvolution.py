import numpy as np
import pytest

from c_spikes.oasis.functions import deconvolve


def _simulate_trace_ar1(g=0.95, sn=0.3, T=500, firerate=0.5, framerate=30, b=0.2, seed=13):
    rng = np.random.default_rng(seed)
    spikes = (rng.random(T) < firerate / float(framerate)).astype(float)
    calcium = spikes.copy()
    for t in range(1, T):
        calcium[t] += g * calcium[t - 1]
    y = b + calcium + sn * rng.standard_normal(T)
    return y.astype(np.double), spikes


def _simulate_trace_ar2(g=(1.7, -0.712), sn=0.3, T=600, firerate=0.5, framerate=30, b=0.2, seed=23):
    rng = np.random.default_rng(seed)
    spikes = (rng.random(T) < firerate / float(framerate)).astype(float)
    calcium = spikes.copy()
    for t in range(2, T):
        calcium[t] += g[0] * calcium[t - 1] + g[1] * calcium[t - 2]
    y = b + calcium + sn * rng.standard_normal(T)
    return y.astype(np.double), spikes


def test_deconvolve_ar1_runs_and_recovers_signal_shape():
    y, spikes = _simulate_trace_ar1()
    c, s, b_hat, g_hat, lam = deconvolve(y, g=(0.95,), sn=0.3, penalty=1)

    assert c.shape == y.shape
    assert s.shape == y.shape
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    assert np.isfinite(float(b_hat))
    assert np.isfinite(float(lam))
    assert np.isfinite(np.asarray(g_hat)).all()

    corr = np.corrcoef(s, spikes)[0, 1]
    assert corr > 0.2


def test_deconvolve_ar2_runs_and_recovers_signal_shape():
    y, spikes = _simulate_trace_ar2()
    c, s, b_hat, g_hat, lam = deconvolve(y, g=(1.7, -0.712), sn=0.3, penalty=1)

    assert c.shape == y.shape
    assert s.shape == y.shape
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    assert np.isfinite(float(b_hat))
    assert np.isfinite(float(lam))
    assert np.isfinite(np.asarray(g_hat)).all()

    corr = np.corrcoef(s, spikes)[0, 1]
    assert corr > 0.1


def test_deconvolve_can_estimate_parameters_when_missing():
    y, _ = _simulate_trace_ar1(seed=99)
    c, s, b_hat, g_hat, lam = deconvolve(y, g=(None,), sn=None, penalty=1)

    assert c.shape == y.shape
    assert s.shape == y.shape
    assert len(np.asarray(g_hat).reshape(-1)) == 1
    assert np.isfinite(float(b_hat))
    assert np.isfinite(float(lam))


def test_deconvolve_rejects_non_float_input():
    y = np.arange(50, dtype=np.int64)
    with pytest.raises(TypeError):
        deconvolve(y, g=(0.95,), sn=0.3)
