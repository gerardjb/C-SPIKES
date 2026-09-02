from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from c_spikes.inference.oasis import OasisConfig, run_oasis_inference
from c_spikes.inference.types import TrialSeries


OASIS_SOURCE_COMMIT = "e738431502040ad7db8f79a12b2927ae9d2f4e7c"


def _config(**overrides) -> OasisConfig:
    values = {
        "dataset_tag": "pytest_oasis",
        "g": (0.9,),
        "sn": 0.05,
        "b": None,
        "b_nonneg": True,
        "optimize_g": 0,
        "penalty": 1,
        "decimate": 1,
        "max_iter": 4,
        "shift": None,
        "window": None,
        "tol": None,
        "discrete_mode": "none",
        "event_threshold": None,
        "threshold_units": "absolute",
        "downsample_label": "raw",
        "uniformity_rtol": 1e-5,
        "uniformity_atol": 1e-10,
        "use_cache": False,
    }
    values.update(overrides)
    return OasisConfig(**values)


def test_unequal_disjoint_trials_are_solved_independently_and_sorted(monkeypatch):
    """Trial boundaries must reset calcium state before outputs are combined."""

    from c_spikes.inference import oasis as oasis_adapter

    late_trial = TrialSeries(
        times=np.asarray([4.0, 4.1, 4.2, 4.3]),
        values=np.asarray([10.0, 11.0, 12.0, 13.0]),
    )
    early_trial = TrialSeries(
        times=np.asarray([1.0, 1.1, 1.2]),
        values=np.asarray([20.0, 21.0, 22.0]),
    )
    calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def fake_deconvolve(y, **kwargs):
        calls.append((np.asarray(y).copy(), dict(kwargs)))
        if len(calls) == 1:
            return (
                np.asarray([1.0, 2.0, 3.0, 4.0]),
                np.asarray([10.0, 11.0, 12.0, 13.0]),
                0.5,
                (1.70, -0.71),
                0.2,
            )
        return (
            np.asarray([5.0, 6.0, 7.0]),
            np.asarray([20.0, 21.0, 22.0]),
            1.5,
            (1.60, -0.63),
            0.3,
        )

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)
    config = _config(
        g=(1.7, -0.712),
        optimize_g=2,
        decimate=2,
        max_iter=7,
        shift=5,
        window=12,
        tol=2e-9,
    )

    result = run_oasis_inference([late_trial, early_trial], config)

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], late_trial.values)
    np.testing.assert_array_equal(calls[1][0], early_trial.values)
    for _, kwargs in calls:
        assert tuple(kwargs["g"]) == pytest.approx(config.g)
        assert kwargs["sn"] == pytest.approx(config.sn)
        assert kwargs["b"] is None
        assert kwargs["b_nonneg"] is True
        assert kwargs["optimize_g"] == 2
        assert kwargs["penalty"] == 1
        assert kwargs["decimate"] == 2
        assert kwargs["max_iter"] == 7
        assert kwargs["shift"] == 5
        assert kwargs["window"] == 12
        assert kwargs["tol"] == pytest.approx(2e-9)

    assert result.name == "oasis"
    np.testing.assert_allclose(result.time_stamps, [1.0, 1.1, 1.2, 4.0, 4.1, 4.2, 4.3])
    np.testing.assert_allclose(result.spike_prob, [20.0, 21.0, 22.0, 10.0, 11.0, 12.0, 13.0])
    np.testing.assert_allclose(result.reconstruction, [6.5, 7.5, 8.5, 1.5, 2.5, 3.5, 4.5])
    assert result.discrete_spikes is None
    assert result.sampling_rate == pytest.approx(10.0)
    assert "discretization" not in result.metadata
    assert "discrete_mode" not in result.metadata["config"]
    assert np.all(np.diff(result.time_stamps) >= 0)

    metadata = result.metadata
    assert metadata["config"]["dataset_tag"] == "pytest_oasis"
    assert metadata["config"]["g"] == pytest.approx([1.7, -0.712])
    assert metadata["config"]["penalty"] == 1
    assert "pytest_oasis" in metadata["cache_tag"]
    assert "cache_key" in metadata
    assert OASIS_SOURCE_COMMIT in metadata["source_version"]

    trial_metadata = metadata["trials"]
    assert len(trial_metadata) == 2
    assert trial_metadata[0]["b"] == pytest.approx(0.5)
    assert np.asarray(trial_metadata[0]["g"]).reshape(-1) == pytest.approx([1.70, -0.71])
    assert trial_metadata[0]["sn"] == pytest.approx(0.05)
    assert trial_metadata[0]["lam"] == pytest.approx(0.2)
    assert trial_metadata[0]["sampling_rate"] == pytest.approx(10.0)
    assert trial_metadata[0]["solver"] == "deconvolve"
    assert trial_metadata[0]["backend"]
    assert "ar2" in trial_metadata[0]["backend"].lower()
    assert trial_metadata[1]["b"] == pytest.approx(1.5)
    assert np.asarray(trial_metadata[1]["g"]).reshape(-1) == pytest.approx([1.60, -0.63])
    assert trial_metadata[1]["sn"] == pytest.approx(0.05)
    assert trial_metadata[1]["lam"] == pytest.approx(0.3)
    assert trial_metadata[1]["sampling_rate"] == pytest.approx(10.0)

    # Adapter plumbing and fake results must not mutate caller-owned trials.
    np.testing.assert_array_equal(late_trial.values, [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(early_trial.values, [20.0, 21.0, 22.0])


def test_absolute_threshold_emits_sorted_binary_support_without_changing_continuous_output(
    monkeypatch,
):
    from c_spikes.inference import oasis as oasis_adapter

    late_trial = TrialSeries(
        times=np.asarray([4.0, 4.1, 4.2, 4.3]),
        values=np.asarray([10.0, 11.0, 12.0, 13.0]),
    )
    early_trial = TrialSeries(
        times=np.asarray([1.0, 1.1, 1.2]),
        values=np.asarray([20.0, 21.0, 22.0]),
    )
    solver_spikes = [
        np.asarray([0.0, 0.3, 0.5, 0.8]),
        np.asarray([0.5, 0.49, 1.0]),
    ]
    calls: list[dict[str, object]] = []

    def fake_deconvolve(values, **kwargs):
        calls.append(dict(kwargs))
        spikes = solver_spikes[len(calls) - 1]
        return np.zeros_like(values), spikes, 0.25, 0.9, 0.1

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)
    result = run_oasis_inference(
        [late_trial, early_trial],
        _config(
            discrete_mode="support",
            event_threshold=0.5,
            threshold_units="absolute",
        ),
    )

    assert len(calls) == 2
    assert all("discrete_mode" not in call for call in calls)
    np.testing.assert_array_equal(
        result.spike_prob,
        np.asarray([0.5, 0.49, 1.0, 0.0, 0.3, 0.5, 0.8]),
    )
    np.testing.assert_array_equal(
        result.discrete_spikes,
        np.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
    )
    assert result.discrete_spikes.dtype == np.uint8
    assert set(np.unique(result.discrete_spikes)) == {0.0, 1.0}
    assert result.metadata["config"]["discrete_mode"] == "support"
    assert result.metadata["config"]["event_threshold"] == pytest.approx(0.5)
    assert result.metadata["config"]["threshold_units"] == "absolute"
    discretization = result.metadata["discretization"]
    assert discretization["semantics"] == "binary_event_support"
    assert discretization["comparison"] == "s >= resolved_threshold"
    assert discretization["event_count"] == 4
    assert discretization["max_events_per_bin"] == 1
    assert [
        trial["discretization"]["resolved_threshold"]
        for trial in result.metadata["trials"]
    ] == pytest.approx([0.5, 0.5])
    assert [
        trial["discretization"]["event_count"]
        for trial in result.metadata["trials"]
    ] == [2, 2]


def test_noise_scaled_threshold_uses_per_trial_fitted_ar2_dominant_decay(monkeypatch):
    from c_spikes.inference import oasis as oasis_adapter

    trials = [
        TrialSeries(
            times=offset + np.arange(4, dtype=np.float64) / 10.0,
            values=np.linspace(0.0, 1.0, 4),
        )
        for offset in (0.0, 2.0)
    ]
    fitted_coefficients = [(1.7, -0.72), (1.25, -0.375)]
    resolved_thresholds = [
        2.0 * 0.4 * np.sqrt(1.0 - 0.9),
        2.0 * 0.4 * np.sqrt(1.0 - 0.75),
    ]
    solver_spikes = [
        np.asarray([0.0, resolved - 1e-5, resolved + 1e-5, resolved + 2e-5])
        for resolved in resolved_thresholds
    ]
    calls = 0

    def fake_deconvolve(values, **_kwargs):
        nonlocal calls
        index = calls
        calls += 1
        return (
            np.zeros_like(values),
            solver_spikes[index],
            0.0,
            fitted_coefficients[index],
            0.1,
        )

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)
    result = run_oasis_inference(
        trials,
        _config(
            g=(1.7, -0.712),
            sn=0.4,
            discrete_mode="support",
            event_threshold=2.0,
            threshold_units="noise_scaled",
        ),
    )

    np.testing.assert_array_equal(
        result.spike_prob,
        np.concatenate(solver_spikes),
    )
    np.testing.assert_array_equal(
        result.discrete_spikes,
        np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
    )
    assert [
        trial["discretization"]["resolved_threshold"]
        for trial in result.metadata["trials"]
    ] == pytest.approx(resolved_thresholds)
    assert result.metadata["discretization"]["threshold_units"] == "noise_scaled"


def test_noise_scaled_threshold_rejects_zero_resolved_noise_scale(monkeypatch):
    from c_spikes.inference import oasis as oasis_adapter

    trial = TrialSeries(
        times=np.arange(4, dtype=np.float64) / 10.0,
        values=np.linspace(0.0, 1.0, 4),
    )

    def fake_deconvolve(values, **_kwargs):
        return np.zeros_like(values), np.zeros_like(values), 0.0, 0.9, 0.1

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)
    with pytest.raises(ValueError, match=r"noise_scaled.*absolute"):
        run_oasis_inference(
            [trial],
            _config(
                sn=0.0,
                discrete_mode="support",
                event_threshold=2.0,
                threshold_units="noise_scaled",
            ),
        )


def test_near_uniform_timestamps_are_accepted_without_resampling(monkeypatch):
    from c_spikes.inference import oasis as oasis_adapter

    times = np.asarray([0.0, 0.1, 0.20000001, 0.30000000, 0.40000001])
    values = np.linspace(0.0, 1.0, times.size)
    observed: list[np.ndarray] = []

    def fake_deconvolve(y, **_kwargs):
        observed.append(np.asarray(y).copy())
        return np.zeros_like(y), np.ones_like(y), 0.25, 0.9, 0.1

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)

    result = run_oasis_inference(
        [TrialSeries(times=times, values=values)],
        _config(uniformity_rtol=2e-6),
    )

    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], values)
    np.testing.assert_array_equal(result.time_stamps, times)
    assert result.sampling_rate == pytest.approx(10.0, rel=2e-6)
    assert result.metadata["trials"][0]["sampling_rate"] == pytest.approx(10.0, rel=2e-6)


def test_default_uniformity_tolerance_accepts_janelia_timestamp_quantization(monkeypatch):
    from c_spikes.inference import oasis as oasis_adapter

    # Bundled Janelia traces alternate between 8.18 ms and 8.20 ms bins after
    # timestamp serialization (about 0.24% relative jitter).
    diffs = np.resize(np.asarray([0.00818, 0.00820]), 15)
    times = np.concatenate(([0.0], np.cumsum(diffs)))
    values = np.sin(np.arange(times.size, dtype=np.float64) / 3.0)

    def fake_deconvolve(y, **_kwargs):
        return np.zeros_like(y), np.ones_like(y), 0.0, 0.9, 0.1

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)
    config = OasisConfig(
        dataset_tag="janelia_quantized",
        g=(0.9,),
        sn=0.05,
        use_cache=False,
    )

    result = run_oasis_inference([TrialSeries(times=times, values=values)], config)

    assert config.uniformity_rtol == pytest.approx(5e-3)
    np.testing.assert_array_equal(result.time_stamps, times)


def test_automatic_parameters_are_estimated_per_trial_and_recorded(monkeypatch):
    from c_spikes.inference import oasis as oasis_adapter

    trials = [
        TrialSeries(
            times=offset + np.arange(14, dtype=np.float64) / 10.0,
            values=baseline + np.sin(np.arange(14, dtype=np.float64) / 3.0),
        )
        for offset, baseline in ((0.0, 1.0), (3.0, 2.0))
    ]
    estimator_calls: list[tuple[np.ndarray, int, float]] = []
    solver_calls: list[dict[str, object]] = []

    def fake_estimate(values, p, fudge_factor):
        estimator_calls.append((np.asarray(values).copy(), p, fudge_factor))
        index = len(estimator_calls)
        return np.asarray([0.90 + index / 100.0]), 0.04 + index / 100.0

    def fake_deconvolve(values, **kwargs):
        solver_calls.append(dict(kwargs))
        return np.zeros_like(values), np.ones_like(values), 0.2, kwargs["g"][0], 0.3

    monkeypatch.setattr(oasis_adapter, "_load_estimate_parameters", lambda: fake_estimate)
    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)

    result = run_oasis_inference(
        trials,
        _config(g=(None,), sn=None, b=0.25, optimize_g=2),
    )

    assert len(estimator_calls) == len(solver_calls) == 2
    for trial, (working_values, order, fudge_factor) in zip(trials, estimator_calls):
        np.testing.assert_allclose(working_values, trial.values - 0.25)
        assert order == 1
        assert fudge_factor == pytest.approx(0.97)
    assert solver_calls[0]["g"] == pytest.approx((0.91,))
    assert solver_calls[0]["sn"] == pytest.approx(0.05)
    assert solver_calls[1]["g"] == pytest.approx((0.92,))
    assert solver_calls[1]["sn"] == pytest.approx(0.06)
    assert result.metadata["trials"][0]["sn"] == pytest.approx(0.05)
    assert result.metadata["trials"][1]["sn"] == pytest.approx(0.06)


@pytest.mark.parametrize(
    "values, g, match",
    [
        (np.linspace(0.0, 1.0, 11), (None,), "at least 12"),
        (np.ones(12), (None,), "non-constant"),
        (np.linspace(0.0, 1.0, 12), (None, None), "at least 13"),
    ],
)
def test_invalid_automatic_estimation_trace_is_rejected_before_native_import(
    monkeypatch, values, g, match
):
    from c_spikes.inference import oasis as oasis_adapter

    monkeypatch.setattr(
        oasis_adapter,
        "_load_estimate_parameters",
        lambda: (_ for _ in ()).throw(AssertionError("estimator loader ran")),
    )
    monkeypatch.setattr(
        oasis_adapter,
        "_load_deconvolve",
        lambda: (_ for _ in ()).throw(AssertionError("solver loader ran")),
    )
    trial = TrialSeries(
        times=np.arange(values.size, dtype=np.float64) / 10.0,
        values=np.asarray(values, dtype=np.float64),
    )

    with pytest.raises(ValueError, match=match):
        run_oasis_inference([trial], _config(g=g, sn=None))


@pytest.mark.parametrize(
    "config_overrides, expected",
    [
        ({"g": (1.0,)}, "AR\\(1\\)"),
        ({"g": (1.2, 0.1)}, "AR\\(2\\)"),
        ({"decimate": None}, "decimate"),
        ({"decimate": 5}, "shortest trial"),
        ({"discrete_mode": "binary"}, "discrete_mode"),
        ({"event_threshold": 0.1}, "event_threshold must be None"),
        ({"threshold_units": "noise_scaled"}, "threshold_units must be 'absolute'"),
        (
            {"discrete_mode": "support", "event_threshold": None},
            "event_threshold must be a positive",
        ),
        (
            {"discrete_mode": "support", "event_threshold": 0.0},
            "event_threshold must be a positive",
        ),
        (
            {"discrete_mode": "support", "event_threshold": np.inf},
            "event_threshold must be a finite",
        ),
        (
            {"discrete_mode": "support", "event_threshold": 0.1, "threshold_units": "relative"},
            "threshold_units",
        ),
    ],
)
def test_invalid_solver_configuration_is_rejected_before_native_import(
    monkeypatch, config_overrides, expected
):
    from c_spikes.inference import oasis as oasis_adapter

    monkeypatch.setattr(
        oasis_adapter,
        "_load_deconvolve",
        lambda: (_ for _ in ()).throw(AssertionError("solver loader ran")),
    )
    trial = TrialSeries(
        times=np.arange(4, dtype=np.float64) / 10.0,
        values=np.linspace(0.0, 1.0, 4),
    )

    with pytest.raises(ValueError, match=expected):
        run_oasis_inference([trial], _config(**config_overrides))


def test_complex_timestamps_are_rejected_without_discarding_imaginary_part(monkeypatch):
    from c_spikes.inference import oasis as oasis_adapter

    monkeypatch.setattr(
        oasis_adapter,
        "_load_deconvolve",
        lambda: (_ for _ in ()).throw(AssertionError("solver loader ran")),
    )
    trial = TrialSeries(
        times=np.asarray([0.0, 0.1 + 0.01j, 0.2, 0.3]),
        values=np.linspace(0.0, 1.0, 4),
    )

    with pytest.raises(ValueError, match="real numeric arrays"):
        run_oasis_inference([trial], _config())


@pytest.mark.parametrize(
    "spikes, fitted_g, expected",
    [
        (np.asarray([0.0, -0.25, 0.0, 0.0]), 0.9, "negative event"),
        (np.zeros(4), 1.1, "AR\\(1\\)"),
    ],
)
def test_invalid_solver_outputs_are_not_cached(monkeypatch, spikes, fitted_g, expected):
    from c_spikes.inference import oasis as oasis_adapter

    def fake_deconvolve(values, **_kwargs):
        return np.zeros_like(values), spikes, 0.0, fitted_g, 0.1

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)
    trial = TrialSeries(
        times=np.arange(4, dtype=np.float64) / 10.0,
        values=np.linspace(0.0, 1.0, 4),
    )

    with pytest.raises(RuntimeError, match=expected):
        run_oasis_inference([trial], _config())


@pytest.mark.parametrize(
    "times",
    [
        pytest.param([0.0, 0.1, 0.25, 0.35], id="irregular"),
        pytest.param([0.0, 0.2, 0.1, 0.3], id="nonmonotonic"),
        pytest.param([0.0, 0.1, 0.1, 0.2], id="repeated"),
    ],
)
def test_invalid_timestamp_spacing_is_rejected_before_inference(monkeypatch, times):
    from c_spikes.inference import oasis as oasis_adapter

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("deconvolution ran for invalid timestamps")

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: must_not_run)
    trial = TrialSeries(
        times=np.asarray(times, dtype=np.float64),
        values=np.linspace(0.0, 1.0, len(times)),
    )

    with pytest.raises(ValueError, match=r"(?i)(timestamp|sampling|uniform|increas)"):
        run_oasis_inference([trial], _config(uniformity_rtol=1e-4))


def test_uniform_trials_with_inconsistent_sampling_rates_are_rejected(monkeypatch):
    from c_spikes.inference import oasis as oasis_adapter

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("deconvolution ran with inconsistent sampling rates")

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: must_not_run)
    trials = [
        TrialSeries(times=np.arange(5) / 10.0, values=np.linspace(0.0, 1.0, 5)),
        TrialSeries(times=2.0 + np.arange(5) / 20.0, values=np.linspace(1.0, 2.0, 5)),
    ]

    with pytest.raises(ValueError, match=r"(?i)(sampling rate|consistent)"):
        run_oasis_inference(trials, _config())


@pytest.mark.parametrize(
    "trials",
    [
        pytest.param([], id="no-trials"),
        pytest.param(
            [TrialSeries(times=np.asarray([]), values=np.asarray([]))],
            id="empty-trial",
        ),
        pytest.param(
            [TrialSeries(times=np.arange(4, dtype=float), values=np.arange(3, dtype=float))],
            id="mismatched-lengths",
        ),
        pytest.param(
            [TrialSeries(times=np.arange(2, dtype=float), values=np.arange(2, dtype=float))],
            id="too-short",
        ),
        pytest.param(
            [TrialSeries(times=np.arange(4, dtype=float).reshape(2, 2), values=np.arange(4.0))],
            id="two-dimensional-times",
        ),
        pytest.param(
            [TrialSeries(times=np.arange(4, dtype=float), values=np.arange(4.0).reshape(2, 2))],
            id="two-dimensional-values",
        ),
        pytest.param(
            [TrialSeries(times=np.asarray([0.0, 0.1, np.inf, 0.3]), values=np.arange(4.0))],
            id="nonfinite-times",
        ),
        pytest.param(
            [TrialSeries(times=np.arange(4, dtype=float), values=np.asarray([0.0, np.nan, 2.0, 3.0]))],
            id="nonfinite-values",
        ),
    ],
)
def test_empty_and_malformed_trials_raise_clear_errors(monkeypatch, trials):
    from c_spikes.inference import oasis as oasis_adapter

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("deconvolution ran for an invalid trial")

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: must_not_run)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        run_oasis_inference(trials, _config())

    assert str(exc_info.value).strip()


def test_inference_package_import_does_not_load_oasis_native_extension():
    """The adapter and unrelated inference methods remain importable without native OASIS."""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    source_root = (repo_root / "src").resolve()
    dependency_paths = [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() != source_root
    ]
    env["PYTHONPATH"] = os.pathsep.join([str(source_root), *dependency_paths])
    env["PYTHONNOUSERSITE"] = "1"
    script = f"""
import sys
from pathlib import Path
import c_spikes
import c_spikes.inference
import c_spikes.inference.oasis
assert Path(c_spikes.__file__).resolve() == Path({str(source_root / 'c_spikes' / '__init__.py')!r})
assert 'c_spikes.oasis.functions' not in sys.modules
assert 'c_spikes.oasis.oasis_methods' not in sys.modules
print('lazy-oasis-import-ok')
"""

    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "lazy-oasis-import-ok" in completed.stdout


@pytest.mark.parametrize(
    "native_error",
    [
        ModuleNotFoundError(
            "No module named 'c_spikes.oasis.oasis_methods'",
            name="c_spikes.oasis.oasis_methods",
        ),
        ValueError("numpy.dtype size changed, may indicate binary incompatibility"),
    ],
    ids=("missing-extension", "stale-numpy-abi"),
)
def test_native_import_failures_have_an_actionable_oasis_error(monkeypatch, native_error):
    import builtins

    import c_spikes.inference as inference_package

    assert inference_package.TrialSeries is TrialSeries

    real_import = builtins.__import__

    def import_without_native(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "c_spikes.oasis.functions":
            raise native_error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_native)
    trial = TrialSeries(
        times=np.arange(4, dtype=np.float64) / 10.0,
        values=np.linspace(0.0, 1.0, 4),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_oasis_inference([trial], _config())

    message = str(exc_info.value).lower()
    assert "oasis" in message
    assert "extension" in message
    assert "build" in message or "install" in message
    assert exc_info.value.__cause__ is native_error
