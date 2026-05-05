from pathlib import Path

import numpy as np

from scripts import code_ocean_inference_demo as demo


def test_inference_driver_uses_method_isolated_producer_runs(tmp_path, monkeypatch):
    edges_8f = {
        demo.JG8F_NOTEBOOK_DATASET: np.array([[465.0, 483.0]], dtype=float),
        demo.JG8F_BIOPHYS_ML_SOURCE_DATASET: np.array([[10.0, 28.0], [30.0, 48.0]], dtype=float),
    }
    edges_8m = {demo.JG8M_NOTEBOOK_DATASET: np.array([[219.0, 237.0]], dtype=float)}

    def fake_load_edges(path: Path):
        token = str(path)
        return edges_8m if "jG8m" in token else edges_8f

    run_batch_calls = []
    import_calls = []
    trialwise_calls = []

    def fake_run_batch(**kwargs):
        run_batch_calls.append(kwargs)

    def fake_import_producer_method(**kwargs):
        import_calls.append(kwargs)

    def fake_trialwise_csv(**kwargs):
        trialwise_calls.append(kwargs)

    monkeypatch.setattr(demo, "_load_edges", fake_load_edges)
    monkeypatch.setattr(demo, "_run_batch", fake_run_batch)
    monkeypatch.setattr(demo, "_import_producer_method", fake_import_producer_method)
    monkeypatch.setattr(demo, "_trialwise_csv", fake_trialwise_csv)
    monkeypatch.setattr(demo, "_filter_finite_correlation_csv", lambda path: path)
    monkeypatch.setattr(demo, "_write_combined_csv", lambda results_dir: Path(results_dir) / "combined.csv")
    monkeypatch.setattr(demo, "_write_parity_tables", lambda **kwargs: Path(kwargs["results_dir"]) / "parity.csv")
    monkeypatch.setattr(demo, "_plot_trace_panel", lambda **kwargs: None)
    monkeypatch.setattr(demo, "_validate_run_methods", lambda **kwargs: None)

    demo.main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--extra-random-epochs",
            "0",
        ]
    )

    producer_calls = [call for call in run_batch_calls if not call.get("eval_only")]
    assert all(len(call["methods"]) == 1 for call in producer_calls)

    producer_runs = {(call["run_tag"], call["methods"][0]) for call in producer_calls}
    assert (demo._producer_run_tag(demo.RUN_JG8F_BASE, "pgas"), "pgas") in producer_runs
    assert (demo._producer_run_tag(demo.RUN_JG8F_BASE, "ens2"), "ens2") in producer_runs
    assert (demo._producer_run_tag(demo.RUN_JG8F_BASE, "cascade"), "cascade") in producer_runs
    assert (demo._producer_run_tag(demo.RUN_JG8F_BASE, "biophys_ml"), "ens2") in producer_runs
    assert (demo._producer_run_tag(demo.RUN_JG8M_BASE, "pgas"), "pgas") in producer_runs
    assert (demo._producer_run_tag(demo.RUN_JG8M_BASE, "ens2"), "ens2") in producer_runs
    assert (demo._producer_run_tag(demo.RUN_JG8M_BASE, "cascade"), "cascade") in producer_runs
    assert (demo._producer_run_tag(demo.RUN_JG8M_BASE, "biophys_ml"), "ens2") in producer_runs

    biophys_imports = [call for call in import_calls if call["target_method"] == "biophys_ml"]
    assert {call["target_run"] for call in biophys_imports} == {demo.RUN_JG8F_BASE, demo.RUN_JG8M_BASE}
    assert all(call["source_method"] == "ens2" for call in biophys_imports)
    assert all(call["cache_tag_prefix"] for call in biophys_imports)

    eval_only = [call for call in run_batch_calls if call.get("eval_only")]
    assert {(call["run_tag"], tuple(call["methods"])) for call in eval_only} == {
        (demo.RUN_JG8F_BASE, ("pgas", "ens2", "cascade", "biophys_ml")),
        (demo.RUN_JG8M_BASE, ("pgas", "ens2", "cascade", "biophys_ml")),
    }

    assert [tuple(call["runs"]) for call in trialwise_calls] == [
        (demo.RUN_JG8F_BASE, demo.RUN_JG8F_PARAMS),
        (demo.RUN_JG8M_BASE,),
    ]
