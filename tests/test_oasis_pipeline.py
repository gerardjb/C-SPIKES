from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest
import scipy.io as sio

import c_spikes.pipeline as pipeline
from c_spikes.inference.types import MethodResult
from c_spikes.pipeline import RunConfig, _build_run_tag, run_batch


def test_defaults_and_existing_run_tags_are_backward_compatible() -> None:
    cfg = RunConfig()

    assert cfg.methods == ("pgas", "ens2", "cascade")
    assert cfg.oasis_discrete_mode == "none"
    assert cfg.oasis_event_threshold is None
    assert cfg.oasis_threshold_units == "absolute"
    assert _build_run_tag(cfg) == "pgasraw_cascadein_ens2"
    assert (
        _build_run_tag(
            RunConfig(
                methods=("pgas",),
                pgas_resample_fs=60.0,
                pgas_maxspikes=4,
                pgas_c0_first_y=True,
            )
        )
        == "pgas60p0_ms4_c0y"
    )
    assert (
        _build_run_tag(
            RunConfig(
                methods=("cascade",),
                cascade_resample_fs=30.0,
                cascade_discretize=False,
            )
        )
        == "cascade30p0_nodisc"
    )
    assert _build_run_tag(RunConfig(methods=("ens2",))) == "ens2"


def test_oasis_run_tag_has_ar_order_and_stable_eight_character_signature() -> None:
    cfg = RunConfig(methods=("oasis",))
    signature_payload = {
        "g": (None,),
        "sn": None,
        "b": None,
        "b_nonneg": True,
        "optimize_g": 0,
        "penalty": 1,
        "decimate": 1,
        "max_iter": None,
        "shift": None,
        "window": None,
        "tol": None,
        "uniformity_rtol": 5e-3,
        "uniformity_atol": 1e-9,
    }
    expected_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]

    assert _build_run_tag(cfg) == f"oasisar1_{expected_signature}"
    assert _build_run_tag(replace(cfg, methods=("OASIS", "oasis"))) == _build_run_tag(cfg)
    assert re.fullmatch(r"oasisar1_[0-9a-f]{8}", _build_run_tag(cfg))
    assert _build_run_tag(replace(cfg, oasis_g=(None, None))).startswith("oasisar2_")


def test_oasis_support_settings_get_a_distinct_run_tag() -> None:
    continuous = RunConfig(methods=("oasis",))
    support = replace(
        continuous,
        oasis_discrete_mode="support",
        oasis_event_threshold=0.5,
    )

    assert _build_run_tag(support) != _build_run_tag(continuous)
    assert _build_run_tag(replace(support, oasis_event_threshold=0.75)) != _build_run_tag(
        support
    )
    assert _build_run_tag(replace(support, oasis_threshold_units="noise_scaled")) != (
        _build_run_tag(support)
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("oasis_g", (0.9,)),
        ("oasis_sn", 0.05),
        ("oasis_b", 0.2),
        ("oasis_b_nonneg", False),
        ("oasis_optimize_g", 2),
        ("oasis_penalty", 0),
        ("oasis_decimate", 2),
        ("oasis_max_iter", 8),
        ("oasis_shift", 4),
        ("oasis_window", 12),
        ("oasis_tol", 1e-8),
        ("oasis_uniformity_rtol", 2e-3),
        ("oasis_uniformity_atol", 2e-9),
    ],
)
def test_each_oasis_setting_invalidates_the_run_tag(
    field_name: str,
    changed_value: object,
) -> None:
    cfg = RunConfig(methods=("oasis",))

    assert _build_run_tag(replace(cfg, **{field_name: changed_value})) != _build_run_tag(cfg)


def _fake_oasis_output(*, with_support: bool = False) -> dict[str, object]:
    result = MethodResult(
        name="oasis",
        time_stamps=np.asarray([0.0, 0.1, 0.2, 0.3]),
        spike_prob=np.asarray([0.0, 0.4, 0.1, 0.0]),
        sampling_rate=10.0,
        metadata={
            "cache_tag": "cell_sraw",
            "cache_key": "0123456789abcdef",
            "config": {"g": [0.9], "penalty": 1},
            "source_version": "oasis-port-test-revision",
            "trials": [
                {
                    "b": 0.2,
                    "g": [0.9],
                    "sn": 0.05,
                    "lam": 0.1,
                    "sampling_rate": 10.0,
                    "backend": "constrained-oasis-ar1",
                }
            ],
        },
        reconstruction=np.asarray([0.2, 0.6, 0.3, 0.2]),
        discrete_spikes=None,
    )
    if with_support:
        discretization = {
            "mode": "support",
            "semantics": "binary_event_support",
            "requested_threshold": 0.25,
            "threshold_units": "absolute",
            "comparison": "s >= resolved_threshold",
            "event_count": 2,
            "max_events_per_bin": 1,
        }
        result.discrete_spikes = np.asarray([0, 1, 1, 0], dtype=np.uint8)
        result.metadata["config"].update(
            {
                "discrete_mode": "support",
                "event_threshold": 0.25,
                "threshold_units": "absolute",
            }
        )
        result.metadata["discretization"] = discretization
        result.metadata["trials"][0]["discretization"] = discretization
    return {
        "methods": {"oasis": result},
        "correlations": {"oasis": 0.75},
        "summary": {
            "downsample_target": "raw",
            "trial_windows_s": [[0.0, 0.3]],
            "epoch_windows_s": [[0.0, 0.3]],
            "epochwise_counts": {
                "gt_count": [1],
                "oasis_samples": [2 if with_support else 0],
            },
            "gt_count": 1,
        },
    }


def test_oasis_batch_forwards_config_and_writes_summary_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_run(dataset_cfg, **_paths):
        observed.append(dataset_cfg)
        return _fake_oasis_output(with_support=True)

    monkeypatch.setattr(pipeline, "run_inference_for_dataset", fake_run)
    cfg = RunConfig(
        data_root=tmp_path / "data",
        datasets=["cell"],
        smoothing_levels=("raw",),
        output_root=tmp_path / "out",
        edges_path=tmp_path / "missing-edges.npy",
        methods=("oasis",),
        use_cache=True,
        oasis_g=(1.7, -0.712),
        oasis_sn=0.08,
        oasis_b=-0.1,
        oasis_b_nonneg=False,
        oasis_optimize_g=2,
        oasis_penalty=0,
        oasis_decimate=3,
        oasis_max_iter=9,
        oasis_shift=4,
        oasis_window=15,
        oasis_tol=1e-8,
        oasis_uniformity_rtol=2e-4,
        oasis_uniformity_atol=3e-10,
        oasis_discrete_mode="support",
        oasis_event_threshold=0.25,
        oasis_threshold_units="absolute",
    )

    summaries = run_batch(cfg)

    assert len(observed) == 1
    forwarded = observed[0]
    assert forwarded.selection.run_pgas is False
    assert forwarded.selection.run_ens2 is False
    assert forwarded.selection.run_cascade is False
    assert forwarded.selection.run_oasis is True
    for field_name in (
        "oasis_g",
        "oasis_sn",
        "oasis_b",
        "oasis_b_nonneg",
        "oasis_optimize_g",
        "oasis_penalty",
        "oasis_decimate",
        "oasis_max_iter",
        "oasis_shift",
        "oasis_window",
        "oasis_tol",
        "oasis_uniformity_rtol",
        "oasis_uniformity_atol",
        "oasis_discrete_mode",
        "oasis_event_threshold",
        "oasis_threshold_units",
    ):
        assert getattr(forwarded, field_name) == getattr(cfg, field_name)

    assert len(summaries) == 1
    summary_path = summaries[0]
    run_tag = _build_run_tag(cfg)
    assert summary_path == cfg.output_root / run_tag / "cell" / "raw" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["methods_run"] == ["oasis"]
    assert summary["oasis_cache"] == {
        "g": [0.9],
        "penalty": 1,
        "discrete_mode": "support",
        "event_threshold": 0.25,
        "threshold_units": "absolute",
    }
    assert summary["oasis_sampling_rate"] == pytest.approx(10.0)
    assert summary["oasis_source_version"] == "oasis-port-test-revision"
    assert summary["oasis_trials"][0]["backend"] == "constrained-oasis-ar1"
    assert summary["oasis_discretization"]["semantics"] == "binary_event_support"
    assert summary["oasis_discretization"]["event_count"] == 2
    assert summary["oasis_samples"] == 2

    discrete = np.load(summary_path.with_name("discrete_spikes.npz"))
    np.testing.assert_array_equal(discrete["oasis"], [0, 1, 1, 0])

    manifest_path = summary_path.with_name("comparison.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_tag"] == run_tag
    assert manifest["methods"] == [
        {
            "label": "oasis",
            "method": "oasis",
            "cache_tag": "cell_sraw",
            "cache_key": "0123456789abcdef",
            "config": {
                "g": [0.9],
                "penalty": 1,
                "discrete_mode": "support",
                "event_threshold": 0.25,
                "threshold_units": "absolute",
            },
            "sampling_rate": 10.0,
            "discretization": summary["oasis_discretization"],
        }
    ]


@pytest.mark.parametrize("with_support", [False, True])
def test_eval_only_loads_oasis_cache_with_optional_discrete_spikes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_support: bool,
) -> None:
    cfg = RunConfig(
        data_root=tmp_path / "data",
        datasets=["cell"],
        smoothing_levels=("raw",),
        output_root=tmp_path / "out",
        edges_path=tmp_path / "missing-edges.npy",
        methods=("oasis",),
        oasis_discrete_mode="support" if with_support else "none",
        oasis_event_threshold=0.25 if with_support else None,
    )
    output = _fake_oasis_output(with_support=with_support)
    monkeypatch.setattr(pipeline, "run_inference_for_dataset", lambda *_args, **_kwargs: output)
    summary_path = run_batch(cfg)[0]

    cache_root = tmp_path / "cache"
    cache_path = cache_root / "oasis" / "cell_sraw" / "0123456789abcdef.mat"
    cache_path.parent.mkdir(parents=True)
    cache_payload = {
        "time_stamps": np.asarray([0.0, 0.1, 0.2, 0.3]),
        "spike_prob": np.asarray([0.0, 0.4, 0.1, 0.0]),
        "reconstruction": np.asarray([0.2, 0.6, 0.3, 0.2]),
    }
    if with_support:
        cache_payload["discrete_spikes"] = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    sio.savemat(cache_path, cache_payload)
    monkeypatch.setattr(pipeline, "get_cache_root", lambda: cache_root)

    import c_spikes.utils as utils

    monkeypatch.setattr(
        utils,
        "load_Janelia_data",
        lambda _path: (
            np.asarray([[0.0, 0.1, 0.2, 0.3]]),
            np.asarray([[0.2, 0.6, 0.3, 0.2]]),
            np.asarray([0.1]),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_inference_for_dataset",
        lambda *_args, **_kwargs: pytest.fail("eval-only must not run inference"),
    )

    eval_summaries = run_batch(replace(cfg, eval_only=True))

    assert eval_summaries == [summary_path]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "oasis" in summary["correlations"]
    expected_count = 2 if with_support else 0
    assert summary["oasis_samples"] == expected_count
    assert summary["epochwise_counts"]["oasis_samples"] == [expected_count]
