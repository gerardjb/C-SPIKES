# TEMPORARY PORT PARITY TESTS:
# Remove after main_cache_noise_port is validated and the port commits are accepted.

from __future__ import annotations

import os
import json

import numpy as np
import pytest
import scipy.io as sio

from c_spikes.inference.cache import (
    load_method_cache,
    load_method_cache_legacy_compatible,
    save_method_cache,
)
from c_spikes.inference.pgas import build_pgas_output_cache_payload, cleanup_pgas_output_dat_files
from c_spikes.inference.types import MethodResult, TrialSeries


pytestmark = [
    pytest.mark.port_parity,
    pytest.mark.skipif(
        os.environ.get("C_SPIKES_RUN_PORT_PARITY") != "1",
        reason="Temporary port parity test; set C_SPIKES_RUN_PORT_PARITY=1 to run.",
    ),
]


def _write_synthetic_pgas_outputs(root, tag: str) -> None:
    traj = root / f"traj_samples_{tag}.dat"
    with traj.open("w", encoding="utf-8") as fh:
        fh.write("index,burst,B,S,C,Y\n")
        for sample in range(3):
            for t in range(4):
                fh.write(
                    f"{sample},{sample % 2},{sample + 0.1 * t:.6f},{sample + t:.6f},"
                    f"{2 * sample + t:.6f},{10 + t:.6f}\n"
                )

    params = root / f"param_samples_{tag}.dat"
    with params.open("w", encoding="utf-8") as fh:
        fh.write("r0,r1,sigma2\n")
        for sample in range(3):
            fh.write(f"{sample + 1:.6f},{sample + 2:.6f},{0.1 * (sample + 1):.6f}\n")

    np.savetxt(root / f"logp_{tag}.dat", np.asarray([-3.0, -1.0, -2.0]))
    (root / f"last_params_{tag}.dat").write_text("keep me\n", encoding="utf-8")


def test_pgas_payload_matches_synthetic_dat_and_cleanup_preserves_last_params(tmp_path):
    dataset_tag = "synthetic_pgas"
    trial_tag = f"{dataset_tag}_trial0"
    _write_synthetic_pgas_outputs(tmp_path, trial_tag)
    trial = TrialSeries(times=np.arange(4, dtype=np.float64), values=np.zeros(4, dtype=np.float64))

    payload = build_pgas_output_cache_payload(
        trials=[trial],
        dataset_tag=dataset_tag,
        output_root=tmp_path,
        burnin=1,
    )
    mat_path = tmp_path / "payload.mat"
    sio.savemat(mat_path, payload, do_compression=True)
    loaded = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["pgas_samples"]

    trajectory = loaded.trajectory_samples
    assert int(trajectory.n_samples) == 3
    assert int(trajectory.n_time) == 4
    np.testing.assert_allclose(trajectory.spikes[2], np.asarray([2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(trajectory.map.spikes, np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert int(loaded.logp.map_sample_index) == 1

    removed = cleanup_pgas_output_dat_files(output_root=tmp_path, dataset_tag=dataset_tag, n_trials=1)
    assert {p.name for p in removed} == {
        f"traj_samples_{trial_tag}.dat",
        f"param_samples_{trial_tag}.dat",
        f"logp_{trial_tag}.dat",
    }
    assert (tmp_path / f"last_params_{trial_tag}.dat").exists()


def test_method_cache_loads_without_embedded_pgas_samples(tmp_path):
    result = MethodResult(
        name="pgas",
        time_stamps=np.asarray([0.0, 1.0]),
        spike_prob=np.asarray([0.0, 1.0]),
        sampling_rate=1.0,
        metadata={"cache_tag": "legacy"},
        reconstruction=np.asarray([0.1, 0.2]),
        discrete_spikes=np.asarray([0.0, 1.0]),
    )
    cfg = {"niter": 1, "burnin": 0}
    save_method_cache("pgas", "legacy", result, cfg, "trace", cache_root=tmp_path)

    loaded = load_method_cache("pgas", "legacy", cfg, "trace", cache_root=tmp_path)
    assert loaded is not None
    np.testing.assert_allclose(loaded.time_stamps, result.time_stamps)
    np.testing.assert_allclose(loaded.spike_prob, result.spike_prob)
    np.testing.assert_allclose(loaded.discrete_spikes, result.discrete_spikes)


def test_legacy_compatible_cache_load_ignores_absolute_path_config_mismatch(tmp_path):
    cache_dir = tmp_path / "pgas" / "cell_sraw_ms2_rsraw_bm0p05"
    cache_dir.mkdir(parents=True)
    mat_path = cache_dir / "legacy.mat"
    sio.savemat(
        mat_path,
        {
            "time_stamps": np.asarray([0.0, 1.0]),
            "spike_prob": np.asarray([0.25, 0.75]),
            "discrete_spikes": np.asarray([0.0, 1.0]),
        },
    )
    legacy_config = {
        "niter": 200,
        "burnin": 100,
        "downsample_target": "raw",
        "constants_file": "/old/worktree/constants.json",
        "gparam_file": "/old/build/20230525_gold.dat",
        "maxspikes": 2,
        "bm_sigma": 0.05,
        "edge_hash": "edge123",
    }
    (cache_dir / "legacy.json").write_text(
        json.dumps(
            {
                "dataset": "cell_sraw_ms2_rsraw_bm0p05",
                "method": "pgas",
                "config": legacy_config,
                "trace_hash": "trace123",
                "sampling_rate": 1.0,
                "metadata": {"cache_tag": "cell_sraw_ms2_rsraw_bm0p05"},
                "cache_key": "legacy",
            }
        ),
        encoding="utf-8",
    )

    current_config = {
        "niter": 200,
        "burnin": 100,
        "downsample_target": "raw",
        "constants_file": "/new/worktree/constants.json",
        "gparam_file": "/new/build/20230525_gold.dat",
        "maxspikes": 2,
        "bm_sigma": 0.05,
        "edge_hash": "edge123",
    }
    loaded = load_method_cache_legacy_compatible(
        "pgas",
        ["cell_sraw_ms2_rsraw_bm0p05"],
        current_config,
        "trace123",
        stable_config_keys=["niter", "burnin", "downsample_target", "maxspikes", "bm_sigma", "edge_hash"],
        cache_root=tmp_path,
    )

    assert loaded is not None
    assert loaded.metadata["cache_style"] == "legacy_path_compatible"
    np.testing.assert_allclose(loaded.discrete_spikes, np.asarray([0.0, 1.0]))

    missed = load_method_cache_legacy_compatible(
        "pgas",
        ["cell_sraw_ms2_rsraw_bm0p05"],
        {**current_config, "edge_hash": "different"},
        "trace123",
        stable_config_keys=["niter", "burnin", "downsample_target", "maxspikes", "bm_sigma", "edge_hash"],
        cache_root=tmp_path,
    )
    assert missed is None
