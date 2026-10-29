import numpy as np

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


def test_deconvolve_can_estimate_parameters_when_missing():
    y, _ = _simulate_trace_ar1(seed=99)
    c, s, b_hat, g_hat, lam = deconvolve(y, g=(None,), sn=None, penalty=1)

    assert c.shape == y.shape
    assert s.shape == y.shape
    assert len(np.asarray(g_hat).reshape(-1)) == 1
    assert np.isfinite(float(b_hat))
    assert np.isfinite(float(lam))
