import numpy as np
import pytest

from c_spikes.oasis.functions import deconvolve
from test_oasis_deconvolution import _simulate_trace_ar1, _simulate_trace_ar2


@pytest.mark.parametrize(
    ("g", "simulate"),
    [
        pytest.param((0.95,), _simulate_trace_ar1, id="ar1"),
        pytest.param((1.7, -0.712), _simulate_trace_ar2, id="ar2"),
    ],
)
def test_l0_and_l1_solutions_obey_the_deconvolution_contract(g, simulate):
    y, _ = simulate()
    l0 = deconvolve(y, g=g, sn=0.3, penalty=0)
    l1 = deconvolve(y, g=g, sn=0.3, penalty=1)

    c0, s0, b0, _, _ = l0
    c1, s1, b1, fitted_g, _ = l1

    assert np.isfinite(c0).all()
    assert np.isfinite(s0).all()
    assert np.isfinite(c1).all()
    assert np.isfinite(s1).all()
    assert np.min(s0) >= -1e-8
    assert np.min(s1) >= -1e-8
    assert np.count_nonzero(s0 > 1e-8) < np.count_nonzero(s1 > 1e-8)
    assert not np.allclose(s0, s1)

    if len(g) == 1:
        recurrence = c1[1:] - fitted_g * c1[:-1]
        np.testing.assert_allclose(s1[1:], recurrence, rtol=1e-8, atol=1e-10)
    else:
        recurrence = c1[2:] - fitted_g[0] * c1[1:-1] - fitted_g[1] * c1[:-2]
        np.testing.assert_allclose(s1[2:], recurrence, rtol=1e-6, atol=5e-4)

    target_rss = 0.3**2 * len(y)
    l0_rss = np.sum((y - (c0 + b0)) ** 2)
    l1_rss = np.sum((y - (c1 + b1)) ** 2)
    assert l0_rss == pytest.approx(target_rss, rel=0.05)
    assert l1_rss == pytest.approx(target_rss, rel=0.05)
