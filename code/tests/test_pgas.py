import numpy as np
import pytest
from c_spikes.utils import load_Janelia_data, spike_times_2_binary
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sample_jg8f_dataset(data_root: Path) -> Path:
    data_dir = data_root / "sample_data" / "janelia_8f" / "excitatory"
    preferred = data_dir / "jGCaMP8f_ANM471993_cell03.mat"
    if preferred.exists():
        return preferred
    candidates = sorted(data_dir.glob("*.mat"))
    if not candidates:
        pytest.skip(f"No jGCaMP8f excitatory sample .mat files found under {data_dir}")
    return candidates[0]


def _gparam_path(repo_root: Path, data_root: Path) -> Path:
    data_copy = data_root / "pgas_parameters" / "20230525_gold.dat"
    if data_copy.exists():
        return data_copy
    return repo_root / "code" / "src" / "c_spikes" / "pgas" / "20230525_gold.dat"


def test_pgas_single_iteration(tmp_path):
    """
    Test PGAS with a single iteration using a sample dataset.
    This test checks if the PGAS runs correctly and produces expected output files
    """
    pgas = pytest.importorskip("c_spikes.pgas.pgas_bound")
    repo_root = _repo_root()
    data_root = repo_root / "data"

    sample_path = _sample_jg8f_dataset(data_root)
    time, data, spike_times = load_Janelia_data(str(sample_path))

    start = 1000 if time.shape[1] >= 2000 else 0
    stop = min(start + 1000, time.shape[1])
    if stop - start < 10:
        pytest.skip(f"Sample dataset is too short for PGAS smoke test: {sample_path}")
    time1 = np.float64(time[0, start:stop]).copy()
    data1 = np.float64(data[0, start:stop]).copy()
    binary_spikes = np.float64(spike_times_2_binary(spike_times, time1))

    analyzer = pgas.Analyzer(
        time=time1,
        data=data1,
        constants_file=str(data_root / "parameter_files" / "constants_GCaMP8_soma.json"),
        output_folder=str(tmp_path),
        column=1,
        tag='unit',
        niter=1,
        append=False,
        verbose=0,
        gtSpikes=binary_spikes,
        has_gtspikes=True,
        maxlen=int(time1.size),
        Gparam_file=str(_gparam_path(repo_root, data_root)),
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
    assert params.shape == (1, 11)
    assert np.isfinite(params).all()

    with open(traj_file) as f:
        header = f.readline().strip()
        assert header == 'index,burst,B,S,C,Y'
        lines = [f.readline().strip() for _ in range(4)]

    for line in lines:
        vals = np.array([float(v) for v in line.split(',')])
        assert vals.size == 6
        assert np.isfinite(vals).all()

    # Sanjeev reports that logp values sometimes come back as NaN especially with low sample rates
    # Haven't had a chance to look into this yet, but for now just assuring that doesn't happen in the unit test
    logp_vals = np.loadtxt(logp_file)
    assert not np.isnan(logp_vals).any()
