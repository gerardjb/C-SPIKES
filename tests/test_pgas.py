"""Small seeded CPU Analyzer schema, ownership and finite-output contract.

This is a two-sweep smoke fixture, not an inference-accuracy benchmark.
"""
from pathlib import Path
import numpy as np
from c_spikes.pgas import pgas_bound_cpu as pgas


def test_analyzer_owned_samples_and_saved_schema(tmp_path):
    fixture = Path(__file__).parent / "fixtures/native_core"
    time = np.array([0., .001, .002]); data = np.zeros(3); spikes = np.array([0.])
    before = tuple(x.tobytes() for x in (time, data, spikes))
    model = pgas.Analyzer(time=time, data=data, constants_file=str(fixture / "analyzer.json"),
        output_folder=str(tmp_path), column=0, tag="test", niter=2, trainedPriorFile="",
        append=False, trim=1, verbose=False, gtSpikes=spikes, has_trained_priors=False,
        has_gtspikes=False, maxlen=0, Gparam_file=str(fixture / "gparams.dat"), seed=17)
    assert model.get_parameter_samples().shape == (0, 0)
    assert model.run() is None
    assert model.get_run_status() == "completed"
    first, second = model.get_parameter_samples(), model.get_parameter_samples()
    assert first.shape == second.shape == (2, 6)
    assert first.dtype == np.float64 and first.flags.owndata and second.flags.owndata
    assert not np.shares_memory(first, second) and np.isfinite(first).all()
    saved = second.tobytes(); first[:] = -123
    assert second.tobytes() == model.get_parameter_samples().tobytes() == saved
    assert tuple(x.tobytes() for x in (time, data, spikes)) == before
    parameter = (tmp_path / "param_samples_test.dat").read_text().splitlines()
    assert parameter[0] == "G_tot,gamma,DCaT,Rf,gam_in,gam_out,sigma2,r0,r1,w01,w10"
    assert len(parameter) == 3 and all(len(row.split(',')) == 11 for row in parameter[1:])
    trajectory = (tmp_path / "traj_samples_test.dat").read_text().splitlines()
    assert trajectory[0] == "index,burst,B,S,C,Y"
    values = np.loadtxt(tmp_path / "traj_samples_test.dat", delimiter=',', skiprows=1, ndmin=2)
    assert values.shape == (6, 6) and np.isfinite(values).all()
    assert np.array_equal(values[:, 0], [0, 0, 0, 1, 1, 1])
    logp = np.loadtxt(tmp_path / "logp_test.dat", ndmin=1)
    assert logp.shape == (2,) and np.isfinite(logp).all()
    assert (tmp_path / "last_params_test.dat").stat().st_size > 0
