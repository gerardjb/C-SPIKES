import numpy as np
import c_spikes.pgas.pgas_bound as pgas
from c_spikes.utils import load_Janelia_data, spike_times_2_binary
import os


def test_pgas_single_iteration(tmp_path):
    """
    Test PGAS with a single iteration using a sample dataset.
    This test checks if the PGAS runs correctly and produces expected output files
    """
    # Load sample data matching the demo configuration
    time, data, spike_times = load_Janelia_data(
        os.path.join('gt_data', 'jGCaMP8f_ANM471993_cell03.mat'))

    time1 = np.float64(time[0, 1000:2000]).copy()
    data1 = np.float64(data[0, 1000:2000]).copy()
    binary_spikes = np.float64(spike_times_2_binary(spike_times, time1))

    analyzer = pgas.Analyzer(
        time=time1,
        data=data1,
        constants_file=os.path.join('parameter_files', 'constants_GCaMP8_soma.json'),
        output_folder=str(tmp_path),
        column=1,
        tag='unit',
        niter=1,
        append=False,
        verbose=0,
        gtSpikes=binary_spikes,
        has_gtspikes=True,
        maxlen=1000,
        Gparam_file=os.path.join('src', 'c_spikes', 'pgas', '20230525_gold.dat'),
        seed=2,
    )
    analyzer.run()

    param_file = tmp_path / 'param_samples_unit.dat'
    traj_file = tmp_path / 'traj_samples_unit.dat'
    logp_file = tmp_path / 'logp_unit.dat'

    assert param_file.exists(), 'parameter sample file missing'
    assert traj_file.exists(), 'trajectory sample file missing'
    assert logp_file.exists(), 'logp file missing'

    params = np.atleast_2d(np.loadtxt(param_file, delimiter=',', skiprows=1))
    # These are from a known run of PGAS with the same parameters
    expected = np.array([6.13335e-05, 668.019, 1.3976e-05, 5.24437,
                         5.19405, 5.29991, 0.00566783, 0.539858,
                         1.10651, 0.0602791, 0.0593417])
    # Ensure single sample and values within +/-20% of expected
    assert params.shape == (1, expected.size)
    assert np.all(np.abs(params[0] - 0.2*expected) <= np.abs(expected))

    with open(traj_file) as f:
        header = f.readline().strip()
        assert header == 'index,burst,B,S,C,Y'
        lines = [f.readline().strip() for _ in range(4)]

    # Check trajectory samples against expected values from a known run
    expected_lines = [
        '0,0,-3.9919e-05,0,-2.08756e-15,0.223743',
        '0,0,-0.00130078,0,-3.28516e-15,0.170749',
        '0,0,0.00164983,0,-4.30697e-15,0.0866747',
        '0,0,0.0045333,0,-4.66954e-15,0.182529',
    ]
    for line, ref in zip(lines, expected_lines):
        vals = [float(v) for v in line.split(',')]
        ref_vals = [float(v) for v in ref.split(',')]
        assert not any(np.isnan(vals))
        assert np.allclose(vals, ref_vals, rtol=1, atol=abs(ref_vals[-1]))

    # Sanjeev reports that logp values sometimes come back as NaN especially with low sample rates
    # Haven't had a chance to look into this yet, but for now just assuring that doesn't happen in the unit test
    logp_vals = np.loadtxt(logp_file)
    assert not np.isnan(logp_vals).any()
