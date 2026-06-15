import numpy as np

from c_spikes.gui import smc_viz


def _write_traj(path, b_samples):
    b_arr = np.asarray(b_samples, dtype=np.float64)
    if b_arr.ndim != 2:
        raise ValueError("b_samples must be sample x time")
    n_samples, time_len = b_arr.shape
    rows = []
    for sample_idx in range(n_samples):
        idx = 0 if sample_idx == 0 else 1
        for time_idx in range(time_len):
            b = float(b_arr[sample_idx, time_idx])
            rows.append([idx, b + 10.0, b, b + 20.0, b + 30.0, b + 40.0])
    np.savetxt(
        path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header="index,burst,B,S,C,Y",
        comments="",
    )


def test_load_traj_stats_creates_reusable_plot_summary(tmp_path, monkeypatch):
    traj_path = tmp_path / "traj_samples_unit_trial0.dat"
    _write_traj(traj_path, [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    stats = smc_viz._load_traj_stats(traj_path, burnin=0)

    assert smc_viz._traj_plot_stats_path(traj_path).exists()
    np.testing.assert_allclose(stats["b_mean"], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(stats["b_std"], [1.0, 1.0, 1.0])
    assert stats["n_samples"] == 2
    assert stats["time_len"] == 3

    def fail_genfromtxt(*_args, **_kwargs):
        raise AssertionError("raw trajectory should not be reparsed")

    monkeypatch.setattr(smc_viz.np, "genfromtxt", fail_genfromtxt)
    cached = smc_viz._load_traj_stats(traj_path, burnin=0)

    np.testing.assert_allclose(cached["b_mean"], stats["b_mean"])
    np.testing.assert_allclose(cached["c_mean"], stats["c_mean"])
    assert cached["burnin_eff"] == stats["burnin_eff"]


def test_load_traj_stats_invalidates_summary_for_burnin(tmp_path):
    traj_path = tmp_path / "traj_samples_unit_trial0.dat"
    _write_traj(traj_path, [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    stats_burnin0 = smc_viz._load_traj_stats(traj_path, burnin=0)
    stats_burnin1 = smc_viz._load_traj_stats(traj_path, burnin=1)

    np.testing.assert_allclose(stats_burnin0["b_mean"], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(stats_burnin1["b_mean"], [3.0, 4.0, 5.0])
    assert stats_burnin1["burnin_eff"] == 1


def test_load_traj_stats_invalidates_summary_when_source_changes(tmp_path):
    traj_path = tmp_path / "traj_samples_unit_trial0.dat"
    _write_traj(traj_path, [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    first = smc_viz._load_traj_stats(traj_path, burnin=0)

    _write_traj(traj_path, [[10.0, 20.0, 30.0], [30.0, 40.0, 50.0]])
    second = smc_viz._load_traj_stats(traj_path, burnin=0)

    np.testing.assert_allclose(first["b_mean"], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(second["b_mean"], [20.0, 30.0, 40.0])
