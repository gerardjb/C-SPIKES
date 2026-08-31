import json

import pytest

from c_spikes import pipeline


@pytest.mark.parametrize(
    ("first_trial_only", "selection_payload", "expected_trial_indices"),
    [
        (False, None, None),
        (True, None, [0]),
        (False, {"cell": [3, 1, 3]}, [1, 3]),
        (True, {"cell": [3, 1, 3]}, [1]),
    ],
)
def test_run_batch_resolves_first_trial_selection(
    tmp_path,
    monkeypatch,
    first_trial_only,
    selection_payload,
    expected_trial_indices,
):
    selection_path = None
    if selection_payload is not None:
        selection_path = tmp_path / "trial_selection.json"
        selection_path.write_text(json.dumps(selection_payload), encoding="utf-8")

    observed_trial_indices = []

    def fake_run_inference_for_dataset(cfg, **_kwargs):
        observed_trial_indices.append(cfg.trial_indices)
        return {
            "methods": {},
            "correlations": {},
            "summary": {"downsample_target": "raw", "gt_count": 0},
        }

    monkeypatch.setattr(
        pipeline,
        "run_inference_for_dataset",
        fake_run_inference_for_dataset,
    )

    cfg = pipeline.RunConfig(
        data_root=tmp_path,
        datasets=["cell"],
        smoothing_levels=["raw"],
        output_root=tmp_path / "output",
        edges_path=tmp_path / "missing_edges.npy",
        methods=(),
        first_trial_only=first_trial_only,
        trial_selection_path=selection_path,
    )

    summaries = pipeline.run_batch(cfg)

    assert observed_trial_indices == [expected_trial_indices]
    assert len(summaries) == 1
