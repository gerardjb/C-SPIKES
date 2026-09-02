from __future__ import annotations

from pathlib import Path

import numpy as np

from c_spikes.gui import inference as gui_inference
from c_spikes.inference.types import MethodResult, TrialSeries


def _settings(**overrides) -> gui_inference.InferenceSettings:
    values = {
        "run_cascade": False,
        "run_ens2": False,
        "run_pgas": False,
        "run_biophys": False,
        "neuron_type": "Exc",
        "use_cache": True,
        "cascade_model_folder": Path("models/cascade"),
        "cascade_model_names": [],
        "ens2_models": [],
        "biophys_models": [],
        "pgas_constants_file": Path("parameters/constants.json"),
        "pgas_gparam_file": Path("parameters/gold.dat"),
    }
    values.update(overrides)
    return gui_inference.InferenceSettings(**values)


def _context(tmp_path: Path) -> gui_inference.RunContext:
    return gui_inference.RunContext(
        data_dir=tmp_path,
        run_tag="run_1",
        run_root=tmp_path / "run_1",
        cache_root=tmp_path / "run_1" / "inference_cache",
        pgas_output_root=tmp_path / "run_1" / "pgas_output",
        pgas_temp_root=tmp_path / "run_1" / "pgas_temp",
    )


def _result(name: str, times: np.ndarray) -> MethodResult:
    return MethodResult(
        name=name,
        time_stamps=np.asarray(times, dtype=np.float64),
        spike_prob=np.zeros(np.asarray(times).size, dtype=np.float64),
        sampling_rate=10.0,
    )


def test_inference_settings_oasis_defaults_preserve_existing_constructors() -> None:
    # This is the pre-OASIS positional constructor shape. New fields must remain
    # defaulted and appended so existing callers continue to work unchanged.
    settings = gui_inference.InferenceSettings(
        False,
        False,
        False,
        False,
        "Exc",
        True,
        Path("models/cascade"),
        [],
        [],
        [],
        Path("parameters/constants.json"),
        Path("parameters/gold.dat"),
    )

    assert settings.run_oasis is False
    assert settings.oasis_g == (None,)
    assert settings.oasis_sn is None
    assert settings.oasis_b is None
    assert settings.oasis_b_nonneg is True
    assert settings.oasis_optimize_g == 0
    assert settings.oasis_penalty == 1
    assert settings.oasis_decimate == 1


def test_normal_path_passes_exact_oasis_config_and_unpadded_trial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    expected = _result("oasis", np.arange(4, dtype=np.float64) / 10.0)

    def fake_oasis(*, trials, config):
        captured["trials"] = trials
        captured["config"] = config
        return expected

    monkeypatch.setattr(gui_inference, "run_oasis_inference", fake_oasis)
    context = _context(tmp_path)
    settings = _settings(
        run_oasis=True,
        use_cache=False,
        oasis_g=(0.91, -0.08),
        oasis_sn=0.17,
        oasis_b=-0.2,
        oasis_b_nonneg=False,
        oasis_optimize_g=3,
        oasis_penalty=0,
        oasis_decimate=4,
    )
    time = np.arange(4, dtype=np.float64) / 10.0
    dff = np.array([0.1, 0.3, -0.2, 0.4], dtype=np.float64)

    results = gui_inference.run_inference_for_epoch(
        epoch_id="epoch_007",
        time=time,
        dff=dff,
        spike_times=None,
        settings=settings,
        context=context,
    )

    assert results == {"oasis": expected}
    trials = captured["trials"]
    assert isinstance(trials, list)
    assert len(trials) == 1
    assert isinstance(trials[0], TrialSeries)
    np.testing.assert_array_equal(trials[0].times, time)
    np.testing.assert_array_equal(trials[0].values, dff)

    config = captured["config"]
    assert config.dataset_tag == "epoch_007"
    assert config.g == (0.91, -0.08)
    assert config.sn == 0.17
    assert config.b == -0.2
    assert config.b_nonneg is False
    assert config.optimize_g == 3
    assert config.penalty == 0
    assert config.decimate == 4
    assert config.downsample_label == "raw"
    assert config.use_cache is False
    assert config.cache_root == context.cache_root


def test_safe_path_scopes_oasis_import_failure_and_runs_other_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    time = np.arange(5, dtype=np.float64) / 20.0
    dff = np.linspace(-0.2, 0.4, time.size)
    expected_pgas = _result("pgas", time)

    def unavailable_oasis(*, trials, config):
        captured["oasis_trials"] = trials
        captured["oasis_config"] = config
        raise RuntimeError(
            "OASIS inference is unavailable because the native extension could not be imported"
        )

    def successful_pgas(*, trials, raw_fs, spike_times, config):
        captured["pgas_trials"] = trials
        return expected_pgas

    monkeypatch.setattr(gui_inference, "run_oasis_inference", unavailable_oasis)
    monkeypatch.setattr(gui_inference, "run_pgas_inference", successful_pgas)
    context = _context(tmp_path)
    settings = _settings(run_oasis=True, run_pgas=True)

    results, errors = gui_inference.run_inference_for_epoch_safe(
        epoch_id="epoch_missing_native",
        time=time,
        dff=dff,
        spike_times=None,
        settings=settings,
        context=context,
    )

    assert results == {"pgas": expected_pgas}
    assert set(errors) == {"oasis"}
    assert "native extension" in errors["oasis"]
    assert captured["pgas_trials"] is captured["oasis_trials"]
    oasis_trials = captured["oasis_trials"]
    assert len(oasis_trials) == 1
    np.testing.assert_array_equal(oasis_trials[0].times, time)
    np.testing.assert_array_equal(oasis_trials[0].values, dff)
    assert captured["oasis_config"].cache_root == context.cache_root
