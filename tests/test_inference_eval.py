from pathlib import Path

import numpy as np
import pytest

from c_spikes.inference import workflow
from c_spikes.inference.eval import build_ground_truth_series
from c_spikes.inference.types import MethodResult


@pytest.mark.parametrize("interval_error", [-5e-10, 5e-10])
def test_ground_truth_grid_is_stable_at_floating_sample_boundary(interval_error):
    start = 7.999999797903001e-05
    sampling_rate = 30.0
    interval_count = 4801
    end = start + (interval_count + interval_error) / sampling_rate
    spike_times = np.asarray([start + 1.0], dtype=np.float64)

    times, trace = build_ground_truth_series(
        spike_times,
        start,
        end,
        reference_fs=sampling_rate,
    )

    assert times.size == interval_count + 1
    assert trace.shape == times.shape
    np.testing.assert_allclose(
        np.diff(times),
        1.0 / sampling_rate,
        rtol=0.0,
        atol=3e-14,
    )
    assert times[0] == start
    assert times[-1] == pytest.approx(
        start + interval_count / sampling_rate,
        rel=0.0,
        abs=3e-14,
    )


@pytest.mark.parametrize(
    ("duration_in_samples", "expected_size", "expected_last_offset"),
    [
        (10.49, 11, 1.0),
        (10.51, 12, 1.1),
    ],
)
def test_ground_truth_grid_uses_nearest_count_without_stretching(
    duration_in_samples,
    expected_size,
    expected_last_offset,
):
    start = 2.0
    sampling_rate = 10.0
    end = start + duration_in_samples / sampling_rate

    times, trace = build_ground_truth_series(
        np.asarray([start + 0.5], dtype=np.float64),
        start,
        end,
        reference_fs=sampling_rate,
    )

    assert times.size == expected_size
    assert trace.shape == times.shape
    assert times[-1] == pytest.approx(start + expected_last_offset)
    assert times[-1] != pytest.approx(end, rel=0.0, abs=1e-6)


def test_workflow_uses_epoch_windows_for_aggregate_and_trialwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    time_stamps = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [2.0, 2.1, 2.2, 2.3, 2.4],
        ],
        dtype=np.float64,
    )
    dff = np.asarray(
        [
            [0.0, 0.1, 1.0, 0.1, 0.0],
            [0.0, 0.2, 0.8, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    spike_times = np.asarray([0.2, 2.2], dtype=np.float64)
    expected_windows = [(0.0, 0.4), (2.0, 2.4)]
    observed = {}

    def fake_cascade_inference(*, trials, config):
        return MethodResult(
            name="cascade",
            time_stamps=np.concatenate([trial.times for trial in trials]),
            spike_prob=np.concatenate([trial.values for trial in trials]),
            sampling_rate=float(config.resample_fs),
            metadata={"input_resample_fs": float(config.resample_fs)},
        )

    def fake_aggregate(methods, spikes, windows, **kwargs):
        observed["aggregate"] = list(windows)
        return {"cascade": 0.25}

    def fake_trialwise(methods, spikes, *, trial_windows, **kwargs):
        observed["trialwise"] = list(trial_windows)
        return {"cascade": [0.2, 0.3]}

    monkeypatch.setattr(workflow, "run_cascade_inference", fake_cascade_inference)
    monkeypatch.setattr(workflow, "compute_correlations_windowed", fake_aggregate)
    monkeypatch.setattr(workflow, "compute_trialwise_correlations_windowed", fake_trialwise)

    config = workflow.DatasetRunConfig(
        dataset_path=tmp_path / "two_epochs.mat",
        selection=workflow.MethodSelection(
            run_pgas=False,
            run_ens2=False,
            run_cascade=True,
        ),
        use_cache=False,
        trialwise_correlations=True,
    )
    result = workflow.run_inference_for_dataset(
        config,
        pgas_constants=tmp_path / "constants",
        pgas_gparam=tmp_path / "gparam",
        pgas_output_root=tmp_path / "pgas",
        ens2_pretrained_root=tmp_path / "ens2",
        cascade_model_root=tmp_path / "cascade",
        dataset_data=(time_stamps, dff, spike_times),
    )

    assert observed == {
        "aggregate": expected_windows,
        "trialwise": expected_windows,
    }
    assert result["correlations"] == {"cascade": 0.25}
    assert result["summary"]["trialwise_correlations"] == {"cascade": [0.2, 0.3]}
