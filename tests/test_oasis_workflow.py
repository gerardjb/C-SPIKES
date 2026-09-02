from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from c_spikes.inference.eval import _preprocess_prediction_for_correlation
from c_spikes.inference.types import MethodResult
from c_spikes.inference.workflow import (
    DatasetRunConfig,
    MethodSelection,
    SmoothingLevel,
    run_inference_for_dataset,
)


def _workflow_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "pgas_constants": tmp_path / "constants.txt",
        "pgas_gparam": tmp_path / "gparam.txt",
        "pgas_output_root": tmp_path / "pgas",
        "ens2_pretrained_root": tmp_path / "ens2",
        "cascade_model_root": tmp_path / "cascade",
    }


def _dataset_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_stamps = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [2.0, 2.1, 2.2, np.nan, np.nan],
        ]
    )
    dff = np.asarray(
        [
            [1.0, 1.1, 1.5, 1.2, 1.0],
            [2.0, 2.6, 2.1, np.nan, np.nan],
        ]
    )
    return time_stamps, dff, np.asarray([0.2, 2.1])


def test_oasis_is_opt_in_and_existing_method_defaults_are_unchanged(tmp_path, monkeypatch):
    from c_spikes.inference import workflow

    defaults = MethodSelection()
    assert defaults.run_pgas is True
    assert defaults.run_ens2 is True
    assert defaults.run_cascade is True
    assert defaults.run_oasis is False

    monkeypatch.setattr(
        workflow,
        "run_oasis_inference",
        lambda *_args, **_kwargs: pytest.fail("default-off OASIS dispatch ran"),
    )
    cfg = DatasetRunConfig(
        dataset_path=tmp_path / "synthetic.mat",
        selection=MethodSelection(run_pgas=False, run_ens2=False, run_cascade=False),
    )

    result = run_inference_for_dataset(
        cfg,
        dataset_data=_dataset_data(),
        **_workflow_paths(tmp_path),
    )

    assert result["methods"] == {}
    assert result["summary"] == {}


def test_workflow_passes_all_oasis_options_and_full_untrimmed_trials(tmp_path, monkeypatch):
    from c_spikes.inference import workflow

    observed = {}

    def fake_run_oasis_inference(*, trials, config):
        observed["trials"] = list(trials)
        observed["config"] = config
        times = np.concatenate([trial.times for trial in trials])
        values = np.concatenate([trial.values for trial in trials])
        order = np.argsort(times, kind="stable")
        return MethodResult(
            name="oasis",
            time_stamps=times[order],
            spike_prob=values[order],
            reconstruction=(values + 10.0)[order],
            sampling_rate=10.0,
            metadata={
                "config": {"penalty": config.penalty, "g": config.g},
                "source_version": "test-source",
                "trials": [{"index": 0}, {"index": 1}],
            },
        )

    monkeypatch.setattr(workflow, "run_oasis_inference", fake_run_oasis_inference)
    cfg = DatasetRunConfig(
        dataset_path=tmp_path / "synthetic.mat",
        smoothing=SmoothingLevel(target_fs=None, label="native.10Hz"),
        edges=np.asarray([[0.1, 0.3], [2.0, 2.1]]),
        selection=MethodSelection(
            run_pgas=False,
            run_ens2=False,
            run_cascade=False,
            run_oasis=True,
        ),
        use_cache=False,
        oasis_g=(1.7, -0.712),
        oasis_sn=0.04,
        oasis_b=-0.2,
        oasis_b_nonneg=False,
        oasis_optimize_g=3,
        oasis_penalty=0,
        oasis_decimate=2,
        oasis_max_iter=8,
        oasis_shift=4,
        oasis_window=16,
        oasis_tol=2e-8,
        oasis_uniformity_rtol=2e-4,
        oasis_uniformity_atol=3e-10,
    )

    result = run_inference_for_dataset(
        cfg,
        dataset_data=_dataset_data(),
        **_workflow_paths(tmp_path),
    )

    # Edges constrain evaluation only: deconvolution receives complete trials.
    assert [trial.times.size for trial in observed["trials"]] == [5, 3]
    np.testing.assert_allclose(observed["trials"][0].times, [0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(observed["trials"][1].times, [2.0, 2.1, 2.2])

    oasis_cfg = observed["config"]
    assert oasis_cfg.dataset_tag == "synthetic"
    assert oasis_cfg.g == pytest.approx((1.7, -0.712))
    assert oasis_cfg.sn == pytest.approx(0.04)
    assert oasis_cfg.b == pytest.approx(-0.2)
    assert oasis_cfg.b_nonneg is False
    assert oasis_cfg.optimize_g == 3
    assert oasis_cfg.penalty == 0
    assert oasis_cfg.decimate == 2
    assert oasis_cfg.max_iter == 8
    assert oasis_cfg.shift == 4
    assert oasis_cfg.window == 16
    assert oasis_cfg.tol == pytest.approx(2e-8)
    assert oasis_cfg.downsample_label == "native.10Hz"
    assert oasis_cfg.uniformity_rtol == pytest.approx(2e-4)
    assert oasis_cfg.uniformity_atol == pytest.approx(3e-10)
    assert oasis_cfg.use_cache is False

    assert set(result["methods"]) == {"oasis"}
    assert result["summary"]["oasis_cache"] == {
        "penalty": 0,
        "g": [1.7, -0.712],
    }
    assert result["summary"]["oasis_sampling_rate"] == pytest.approx(10.0)
    assert result["summary"]["oasis_source_version"] == "test-source"
    assert result["summary"]["oasis_trials"] == [{"index": 0}, {"index": 1}]


def test_oasis_only_synthetic_workflow_keeps_evaluation_and_reconstruction(
    tmp_path, monkeypatch
):
    from c_spikes.inference import workflow

    def fake_run_oasis_inference(*, trials, config):
        assert [trial.values.size for trial in trials] == [5, 3]
        times = np.concatenate([trial.times for trial in trials])
        spikes = np.concatenate(
            [np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])]
        )
        reconstruction = np.concatenate([trial.values + 0.5 for trial in trials])
        order = np.argsort(times, kind="stable")
        return MethodResult(
            name="oasis",
            time_stamps=times[order],
            spike_prob=spikes[order],
            reconstruction=reconstruction[order],
            discrete_spikes=None,
            sampling_rate=10.0,
            metadata={
                "config": {"penalty": config.penalty},
                "source_version": "test-source",
                "trials": [{"length": 5}, {"length": 3}],
            },
        )

    monkeypatch.setattr(workflow, "run_oasis_inference", fake_run_oasis_inference)
    cfg = DatasetRunConfig(
        dataset_path=tmp_path / "synthetic.mat",
        selection=MethodSelection(
            run_pgas=False,
            run_ens2=False,
            run_cascade=False,
            run_oasis=True,
        ),
        use_cache=False,
        oasis_g=(0.9,),
        oasis_sn=0.05,
    )

    result = run_inference_for_dataset(
        cfg,
        dataset_data=_dataset_data(),
        **_workflow_paths(tmp_path),
    )

    oasis_result = result["methods"]["oasis"]
    assert oasis_result.discrete_spikes is None
    np.testing.assert_allclose(
        oasis_result.reconstruction,
        [1.5, 1.6, 2.0, 1.7, 1.5, 2.5, 3.1, 2.6],
    )
    assert "oasis" in result["correlations"]
    assert np.isfinite(result["correlations"]["oasis"])
    assert result["summary"]["epochwise_counts"]["oasis_samples"] == [0, 0]
    np.testing.assert_array_equal(result["spike_times"]["oasis"], np.asarray([]))


def test_oasis_predictions_do_not_receive_the_pgas_one_bin_shift():
    values = np.asarray([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(_preprocess_prediction_for_correlation("oasis", values), values)
    np.testing.assert_array_equal(
        _preprocess_prediction_for_correlation("pgas", values),
        [2.0, 3.0, 0.0],
    )
