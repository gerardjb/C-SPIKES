from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np

import c_spikes.inference.oasis as oasis_adapter
from c_spikes.inference.oasis import OasisConfig, run_oasis_inference
from c_spikes.inference.types import TrialSeries


def _trials() -> list[TrialSeries]:
    return [
        TrialSeries(
            times=np.arange(8, dtype=np.float64) / 10.0,
            values=np.asarray([0.2, 0.3, 0.8, 0.5, 0.4, 0.9, 0.5, 0.3], dtype=np.float64),
        ),
        TrialSeries(
            times=2.0 + np.arange(6, dtype=np.float64) / 10.0,
            values=np.asarray([0.4, 0.7, 0.5, 1.0, 0.6, 0.4], dtype=np.float64),
        ),
    ]


def _config(cache_root, **overrides) -> OasisConfig:
    values = {
        "dataset_tag": "cache_contract",
        "g": (1.7, -0.712),
        "sn": 0.05,
        "b": None,
        "b_nonneg": True,
        "optimize_g": 0,
        "penalty": 1,
        "decimate": 1,
        "max_iter": None,
        "shift": None,
        "window": None,
        "tol": None,
        "downsample_label": "raw",
        "uniformity_rtol": 1e-3,
        "uniformity_atol": 1e-9,
        "use_cache": True,
        "cache_root": cache_root,
    }
    values.update(overrides)
    return OasisConfig(**values)


def _install_fake_solver(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_deconvolve(values, *args, **kwargs):
        values = np.asarray(values, dtype=np.float64)
        calls.append(
            {
                "values": values.copy(),
                "args": args,
                "kwargs": dict(kwargs),
            }
        )
        baseline_arg = kwargs.get("b")
        baseline = 0.2 if baseline_arg is None else float(baseline_arg)
        coefficients = kwargs.get("g", (1.7, -0.712))
        coefficients = tuple(float(value) for value in np.asarray(coefficients).reshape(-1))
        fitted_g = coefficients[0] if len(coefficients) == 1 else coefficients
        calcium = values - baseline
        spikes = np.maximum(values - values[0], 0.0)
        return calcium, spikes, baseline, fitted_g, 0.025

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", lambda: fake_deconvolve)
    return calls


def _assert_results_equal(actual, expected) -> None:
    assert actual.name == expected.name == "oasis"
    assert actual.sampling_rate == expected.sampling_rate
    np.testing.assert_array_equal(actual.time_stamps, expected.time_stamps)
    np.testing.assert_array_equal(actual.spike_prob, expected.spike_prob)
    np.testing.assert_array_equal(actual.reconstruction, expected.reconstruction)
    assert actual.discrete_spikes is None
    assert expected.discrete_spikes is None
    actual_metadata = dict(actual.metadata)
    expected_metadata = dict(expected.metadata)
    assert actual_metadata.pop("cache_hit") is True
    assert expected_metadata.pop("cache_hit") is False
    assert actual_metadata == expected_metadata


def test_identical_call_hits_shared_cache_without_loading_solver_and_round_trips_metadata(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "inference_cache"
    config = _config(cache_root)
    calls = _install_fake_solver(monkeypatch)

    first = run_oasis_inference(_trials(), config)

    assert len(calls) == 2
    assert first.metadata["cache_tag"] == f"{config.dataset_tag}_sraw"
    assert first.metadata["cache_key"]
    assert first.metadata["source_version"]
    assert len(first.metadata["trials"]) == 2

    meta_path = (
        cache_root
        / "oasis"
        / first.metadata["cache_tag"]
        / f"{first.metadata['cache_key']}.json"
    )
    mat_path = meta_path.with_suffix(".mat")
    assert meta_path.is_file()
    assert mat_path.is_file()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))

    assert payload["method"] == "oasis"
    assert payload["dataset"] == first.metadata["cache_tag"]
    assert payload["cache_key"] == first.metadata["cache_key"]
    assert payload["metadata"] == first.metadata
    assert "cspikes-numerical-v1" in payload["config"]["source_version"]
    for name in (
        "g",
        "sn",
        "b",
        "b_nonneg",
        "optimize_g",
        "penalty",
        "decimate",
        "max_iter",
        "shift",
        "window",
        "tol",
        "downsample_label",
        "uniformity_rtol",
        "uniformity_atol",
    ):
        assert name in payload["config"]

    def fail_if_native_solver_is_loaded():
        raise AssertionError("a cache hit must not load the native OASIS solver")

    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", fail_if_native_solver_is_loaded)
    second = run_oasis_inference(_trials(), config)

    assert len(calls) == 2
    _assert_results_equal(second, first)


def test_unreadable_cache_does_not_prevent_inference(tmp_path, monkeypatch):
    config = _config(tmp_path / "inference_cache")
    calls = _install_fake_solver(monkeypatch)
    monkeypatch.setattr(
        oasis_adapter,
        "load_method_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only cache")),
    )

    result = run_oasis_inference(_trials(), config)

    assert result.name == "oasis"
    assert result.metadata["cache_hit"] is False
    assert len(calls) == 2


def test_automatic_parameter_cache_hit_loads_neither_estimator_nor_solver(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "inference_cache"
    config = _config(cache_root, g=(None, None), sn=None)
    trials = [
        TrialSeries(
            times=offset + np.arange(14, dtype=np.float64) / 10.0,
            values=baseline + np.sin(np.arange(14, dtype=np.float64) / 3.0),
        )
        for offset, baseline in ((0.0, 1.0), (3.0, 2.0))
    ]
    estimator_calls: list[np.ndarray] = []

    def fake_estimate(values, p, fudge_factor):
        estimator_calls.append(np.asarray(values).copy())
        assert p == 2
        assert fudge_factor == 0.98
        return np.asarray([1.7, -0.712]), 0.05

    monkeypatch.setattr(oasis_adapter, "_load_estimate_parameters", lambda: fake_estimate)
    solver_calls = _install_fake_solver(monkeypatch)

    first = run_oasis_inference(trials, config)
    assert len(estimator_calls) == len(solver_calls) == 2

    def fail_loader():
        raise AssertionError("an automatic-parameter cache hit must not load native OASIS")

    monkeypatch.setattr(oasis_adapter, "_load_estimate_parameters", fail_loader)
    monkeypatch.setattr(oasis_adapter, "_load_deconvolve", fail_loader)

    second = run_oasis_inference(trials, config)

    _assert_results_equal(second, first)


def test_every_exposed_config_field_has_cache_invalidation_coverage(tmp_path, monkeypatch):
    cache_root = tmp_path / "inference_cache"
    base = _config(cache_root)
    calls = _install_fake_solver(monkeypatch)
    trials = _trials()

    run_oasis_inference(trials, base)
    assert len(calls) == len(trials)

    variants = {
        "dataset_tag": replace(base, dataset_tag="cache_contract_other"),
        "g": replace(base, g=(1.65, -0.68)),
        "sn": replace(base, sn=0.06),
        "b": replace(base, b=0.1),
        "b_nonneg": replace(base, b_nonneg=False),
        "optimize_g": replace(base, optimize_g=1),
        "penalty": replace(base, penalty=0),
        "decimate": replace(base, decimate=2),
        "max_iter": replace(base, max_iter=25),
        "shift": replace(base, shift=4),
        "window": replace(base, window=6),
        "tol": replace(base, tol=1e-8),
        "downsample_label": replace(base, downsample_label="downsampled"),
        "uniformity_rtol": replace(base, uniformity_rtol=2e-3),
        "uniformity_atol": replace(base, uniformity_atol=2e-9),
        "use_cache": replace(base, use_cache=False),
        "cache_root": replace(base, cache_root=tmp_path / "alternate_cache"),
    }
    assert set(variants) == {field.name for field in fields(OasisConfig)}

    for field_name, variant in variants.items():
        calls_before = len(calls)
        run_oasis_inference(trials, variant)
        assert len(calls) == calls_before + len(trials), field_name


def test_trace_content_and_timestamps_invalidate_cache(tmp_path, monkeypatch):
    config = _config(tmp_path / "inference_cache")
    calls = _install_fake_solver(monkeypatch)
    original = _trials()

    run_oasis_inference(original, config)
    assert len(calls) == 2

    changed_values = _trials()
    changed_values[0].values = changed_values[0].values.copy()
    changed_values[0].values[3] += 0.125
    run_oasis_inference(changed_values, config)
    assert len(calls) == 4

    changed_times = _trials()
    changed_times[0].times = changed_times[0].times + 0.01
    run_oasis_inference(changed_times, config)
    assert len(calls) == 6


def test_identical_flattened_arrays_with_different_trial_boundaries_do_not_share_cache(
    tmp_path, monkeypatch
):
    config = _config(tmp_path / "inference_cache")
    calls = _install_fake_solver(monkeypatch)
    times = np.arange(8, dtype=np.float64) / 10.0
    values = np.asarray([0.2, 0.4, 0.8, 0.5, 0.3, 0.9, 0.6, 0.4], dtype=np.float64)
    four_plus_four = [
        TrialSeries(times=times[:4], values=values[:4]),
        TrialSeries(times=times[4:], values=values[4:]),
    ]
    three_plus_five = [
        TrialSeries(times=times[:3], values=values[:3]),
        TrialSeries(times=times[3:], values=values[3:]),
    ]

    np.testing.assert_array_equal(
        np.concatenate([trial.times for trial in four_plus_four]),
        np.concatenate([trial.times for trial in three_plus_five]),
    )
    np.testing.assert_array_equal(
        np.concatenate([trial.values for trial in four_plus_four]),
        np.concatenate([trial.values for trial in three_plus_five]),
    )

    run_oasis_inference(four_plus_four, config)
    assert len(calls) == 2

    run_oasis_inference(three_plus_five, config)
    assert len(calls) == 4

    run_oasis_inference(three_plus_five, config)
    assert len(calls) == 4
