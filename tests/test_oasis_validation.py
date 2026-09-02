import warnings

import numpy as np
import pytest

from c_spikes.oasis.functions import deconvolve, estimate_time_constant


def _simulate_trace(g, *, baseline=0.4, noise=0.02, length=120, seed=41):
    """Return a short deterministic trace with several well-separated events."""

    spikes = np.zeros(length, dtype=np.float64)
    event_times = np.arange(7, length, 19)
    spikes[event_times] = 0.7 + 0.1 * (np.arange(event_times.size) % 4)

    calcium = np.zeros(length, dtype=np.float64)
    for index in range(length):
        calcium[index] = spikes[index]
        if index >= 1:
            calcium[index] += g[0] * calcium[index - 1]
        if index >= 2 and len(g) == 2:
            calcium[index] += g[1] * calcium[index - 2]

    rng = np.random.default_rng(seed)
    return baseline + calcium + noise * rng.standard_normal(length)


def _assert_clear_validation_error(y, **kwargs):
    with pytest.raises((TypeError, ValueError)) as exc_info:
        deconvolve(y, **kwargs)
    assert str(exc_info.value).strip()


@pytest.mark.parametrize(
    "y",
    [
        pytest.param([0.1, 0.2, 0.4, 0.3, 0.2], id="float-list"),
        pytest.param(np.linspace(0.1, 0.5, 24, dtype=np.float32), id="float32-array"),
    ],
)
def test_deconvolve_accepts_float_inputs_and_returns_float64(y):
    c, s, b_hat, g_hat, lam = deconvolve(y, g=(0.9,), sn=0.05, penalty=1)

    assert c.dtype == np.float64
    assert s.dtype == np.float64
    assert c.shape == np.asarray(y).shape
    assert s.shape == np.asarray(y).shape
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    assert np.isfinite(float(b_hat))
    assert np.isfinite(np.asarray(g_hat, dtype=np.float64)).all()
    assert np.isfinite(float(lam))


@pytest.mark.parametrize(
    "y",
    [
        pytest.param(np.arange(12, dtype=np.int64), id="integer-array"),
        pytest.param(list(range(12)), id="integer-list"),
    ],
)
def test_deconvolve_rejects_integer_input(y):
    _assert_clear_validation_error(y, g=(0.9,), sn=0.05)


@pytest.mark.parametrize(
    "y",
    [
        pytest.param([], id="empty-list"),
        pytest.param(np.array([], dtype=np.float64), id="empty-array"),
        pytest.param(0.5, id="python-scalar"),
        pytest.param(np.array(0.5), id="zero-dimensional-array"),
        pytest.param(np.ones((4, 3), dtype=np.float64), id="two-dimensional-array"),
    ],
)
def test_deconvolve_rejects_empty_or_non_vector_input(y):
    _assert_clear_validation_error(y, g=(0.9,), sn=0.05)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_deconvolve_rejects_nonfinite_trace_values(bad_value):
    y = np.linspace(0.1, 0.5, 24, dtype=np.float64)
    y[8] = bad_value

    _assert_clear_validation_error(y, g=(0.9,), sn=0.05)


@pytest.mark.parametrize(
    "g",
    [
        pytest.param((), id="empty"),
        pytest.param((0.8, -0.1, 0.01), id="order-three"),
        pytest.param((None, -0.1), id="mixed-none-first"),
        pytest.param((0.9, None), id="mixed-none-second"),
        pytest.param((np.nan,), id="nonfinite-ar1"),
        pytest.param((0.0,), id="zero-ar1"),
        pytest.param((1.0,), id="unit-ar1"),
        pytest.param((0.5, 0.2), id="negative-ar2-root"),
    ],
)
def test_deconvolve_rejects_unsupported_or_partially_missing_g(g):
    y = np.linspace(0.1, 0.5, 24, dtype=np.float64)
    _assert_clear_validation_error(y, g=g, sn=0.05)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        pytest.param("sn", -0.01, id="negative-sn"),
        pytest.param("sn", np.nan, id="nan-sn"),
        pytest.param("sn", np.inf, id="infinite-sn"),
        pytest.param("sn", "low", id="nonnumeric-sn"),
        pytest.param("sn", True, id="boolean-sn"),
        pytest.param("b", np.nan, id="nan-baseline"),
        pytest.param("b", np.inf, id="infinite-baseline"),
        pytest.param("b", "auto", id="nonnumeric-baseline"),
        pytest.param("b", False, id="boolean-baseline"),
    ],
)
def test_deconvolve_rejects_invalid_noise_or_baseline(parameter, value):
    y = np.linspace(0.1, 0.5, 24, dtype=np.float64)
    kwargs = {"g": (0.9,), "sn": 0.05, parameter: value}
    _assert_clear_validation_error(y, **kwargs)


@pytest.mark.parametrize("penalty", [-1, 2, 0.5, True, "1"])
def test_deconvolve_rejects_invalid_penalty(penalty):
    y = np.linspace(0.1, 0.5, 24, dtype=np.float64)
    _assert_clear_validation_error(y, g=(0.9,), sn=0.05, penalty=penalty)


@pytest.mark.parametrize("optimize_g", [-1, 0.5, True, "1"])
def test_deconvolve_rejects_invalid_optimize_g(optimize_g):
    y = np.linspace(0.1, 0.5, 24, dtype=np.float64)
    _assert_clear_validation_error(y, g=(0.9,), sn=0.05, optimize_g=optimize_g)


@pytest.mark.parametrize("decimate", [-1, 0, 1.5, True, "2"])
def test_deconvolve_rejects_invalid_decimation_factor(decimate):
    y = np.linspace(0.1, 0.5, 24, dtype=np.float64)
    _assert_clear_validation_error(y, g=(0.9,), sn=0.05, decimate=decimate)


@pytest.mark.parametrize(
    ("g", "baseline", "solver_kwargs"),
    [
        pytest.param((0.92,), 1.25, {}, id="ar1-positive"),
        pytest.param((0.92,), -0.25, {}, id="ar1-negative"),
        pytest.param((1.7, -0.712), 1.25, {"window": 36, "shift": 18}, id="ar2-positive"),
        pytest.param((1.7, -0.712), -0.25, {"window": 36, "shift": 18}, id="ar2-negative"),
    ],
)
def test_deconvolve_honors_explicit_baseline_for_ar1_and_ar2(g, baseline, solver_kwargs):
    y = _simulate_trace(g, baseline=baseline, length=108)

    actual = deconvolve(
        y,
        g=g,
        sn=0.02,
        b=baseline,
        penalty=1,
        decimate=1,
        **solver_kwargs,
    )
    reference = deconvolve(
        y - baseline,
        g=g,
        sn=0.02,
        b=0.0,
        penalty=1,
        decimate=1,
        **solver_kwargs,
    )
    c, s, b_hat, fitted_g, lam = actual

    assert b_hat == pytest.approx(baseline)
    assert c.shape == y.shape
    assert s.shape == y.shape
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    np.testing.assert_allclose(c, reference[0])
    np.testing.assert_allclose(s, reference[1])
    np.testing.assert_allclose(fitted_g, reference[3])
    assert lam == pytest.approx(reference[4])


def test_ar2_optimize_g_accepts_tuple_input():
    g = (1.7, -0.712)
    y = _simulate_trace(g, length=90)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = deconvolve(
            y,
            g=g,
            sn=0.02,
            optimize_g=1,
            penalty=1,
            decimate=5,
            window=30,
            shift=15,
        )

    assert result[0].shape == y.shape
    assert result[1].shape == y.shape
    assert np.isfinite(result[0]).all()
    assert np.isfinite(result[1]).all()


def test_ar2_optimize_g_does_not_mutate_list_input():
    g = [1.7, -0.712]
    original_g = g.copy()
    y = _simulate_trace(g, length=90)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        list_result = deconvolve(
            y,
            g=g,
            sn=0.02,
            optimize_g=1,
            penalty=1,
            decimate=5,
            window=30,
            shift=15,
        )
        tuple_result = deconvolve(
            y,
            g=tuple(original_g),
            sn=0.02,
            optimize_g=1,
            penalty=1,
            decimate=5,
            window=30,
            shift=15,
        )

    assert g == original_g
    for list_value, tuple_value in zip(list_result, tuple_result):
        np.testing.assert_allclose(list_value, tuple_value)


@pytest.mark.parametrize(
    ("seed", "order", "expected"),
    [
        pytest.param(2, 1, [0.15], id="ar1-repaired-root"),
        pytest.param(0, 2, [0.4836518095638108, -0.05004777143457162], id="ar2"),
    ],
)
def test_estimate_time_constant_is_deterministic_without_using_global_rng(seed, order, expected):
    y = np.random.default_rng(seed).normal(size=100)
    original_state = np.random.get_state()

    try:
        np.random.seed(1729)
        expected_state = np.random.get_state()

        first = estimate_time_constant(y, p=order, sn=0.1, lags=10)
        second = estimate_time_constant(y, p=order, sn=0.1, lags=10)

        actual_state = np.random.get_state()
        assert actual_state[0] == expected_state[0]
        np.testing.assert_array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, expected, rtol=0, atol=1e-15)
    finally:
        np.random.set_state(original_state)


@pytest.mark.parametrize(
    ("y", "g"),
    [
        pytest.param(np.linspace(0.0, 1.0, 11), (None,), id="short-ar1"),
        pytest.param(np.linspace(0.0, 1.0, 12), (None, None), id="short-ar2"),
        pytest.param(np.ones(32), (None,), id="constant-ar1"),
        pytest.param(np.ones(32), (None, None), id="constant-ar2"),
    ],
)
def test_deconvolve_rejects_short_or_constant_parameter_estimation(y, g):
    _assert_clear_validation_error(y, g=g, sn=None)


@pytest.mark.parametrize(
    "g, length, seed",
    [
        pytest.param((None,), 12, 0, id="minimum-ar1"),
        pytest.param((None, None), 13, 1, id="minimum-ar2"),
    ],
)
def test_minimum_length_automatic_estimation_is_warning_free_and_deterministic(g, length, seed):
    y = np.random.default_rng(seed).normal(0.3, 0.1, length)

    first = deconvolve(y, g=g, sn=None, penalty=1, decimate=1)
    second = deconvolve(y, g=g, sn=None, penalty=1, decimate=1)

    for first_value, second_value in zip(first, second):
        np.testing.assert_allclose(first_value, second_value, rtol=0, atol=0)
        assert np.isfinite(np.asarray(first_value, dtype=np.float64)).all()


def test_ar1_nondivisible_decimation_returns_full_finite_output():
    g = (0.92,)
    y = _simulate_trace(g, length=103)

    c, s, _, _, _ = deconvolve(
        y,
        g=g,
        sn=0.02,
        b=0.4,
        penalty=1,
        decimate=5,
    )

    assert c.shape == y.shape
    assert s.shape == y.shape
    assert c.dtype == np.float64
    assert s.dtype == np.float64
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()


def test_ar1_l0_retries_full_resolution_when_decimated_warm_start_is_infeasible():
    y = np.random.default_rng(6).normal(0.2, 0.3, 10)

    actual = deconvolve(y, g=(0.92,), sn=0.05, penalty=0, decimate=5)
    expected = deconvolve(y, g=(0.92,), sn=0.05, penalty=0, decimate=1)

    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=0, atol=0)
    c, s, b_hat, fitted_g, lam = actual
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    assert np.min(c) >= -1e-12
    assert np.min(s) >= -1e-12
    assert np.isfinite(b_hat)
    assert np.isfinite(lam)
    assert fitted_g == pytest.approx(0.92)
    np.testing.assert_allclose(s[1:], c[1:] - fitted_g * c[:-1], atol=1e-12)


def test_ar2_l0_infeasible_sparsification_retains_a_valid_solution():
    y = np.random.default_rng(2).normal(0.2, 0.3, 20)
    kwargs = {"g": (1.7, -0.72), "sn": 0.05, "penalty": 0, "decimate": 2}

    first = deconvolve(y, **kwargs)
    second = deconvolve(y, **kwargs)

    for first_value, second_value in zip(first, second):
        np.testing.assert_allclose(first_value, second_value, rtol=0, atol=0)
    c, s, b_hat, fitted_g, lam = first
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    assert np.min(c) >= -1e-12
    assert np.min(s) >= -1e-12
    assert np.isfinite(b_hat)
    assert np.isfinite(lam)
    np.testing.assert_allclose(fitted_g, (1.7, -0.72))
    np.testing.assert_allclose(
        s[2:],
        c[2:] - fitted_g[0] * c[1:-1] - fitted_g[1] * c[:-2],
        rtol=1e-7,
        atol=1e-9,
    )


def test_ar2_l0_event_at_first_sample_does_not_evaluate_empty_prefix():
    y = np.random.default_rng(0).normal(0.2, 0.3, 20)

    c, s, _, fitted_g, _ = deconvolve(
        y,
        g=(1.7, -0.72),
        sn=0.001,
        penalty=0,
        decimate=1,
    )

    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    assert np.min(c) >= -1e-12
    assert np.min(s) >= -1e-12
    np.testing.assert_allclose(
        s[2:],
        c[2:] - fitted_g[0] * c[1:-1] - fitted_g[1] * c[:-2],
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.parametrize("window, shift", [(50, 100), (None, 300)])
def test_ar2_onnls_limits_block_shift_to_effective_window(window, shift):
    y = np.random.default_rng(17).normal(0.2, 0.1, 220)

    c, s, _, _, _ = deconvolve(
        y,
        g=(1.7, -0.72),
        sn=0.1,
        b=0.0,
        penalty=1,
        decimate=1,
        window=window,
        shift=shift,
    )

    assert c.shape == y.shape
    assert s.shape == y.shape
    assert np.isfinite(c).all()
    assert np.isfinite(s).all()
    assert np.min(s) >= -1e-12


def test_deconvolve_rejects_decimation_larger_than_trace():
    y = np.linspace(0.1, 0.5, 20, dtype=np.float64)
    _assert_clear_validation_error(y, g=(0.9,), sn=0.05, decimate=21)
