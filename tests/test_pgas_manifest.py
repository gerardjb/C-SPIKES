"""Public manifest/CLI tests; no native extension is loaded or GPU requested."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux" or sys.version_info < (3, 9),
                                reason="Launcher requires Linux and Python 3.9+")

from c_spikes.pgas import pool
from c_spikes.pgas.manifest import prepare_manifest


@pytest.fixture
def spec(tmp_path):
    np.savez(tmp_path/'window.npz', time=np.arange(5)*.01, fluorescence=np.ones(5))
    (tmp_path/'constants.json').write_text(json.dumps({'MCMC': {'seed': 5, 'nparticles': 1000}}))
    (tmp_path/'sensor.dat').write_text('parameters')
    binary = tmp_path/'fake-native.so'; binary.write_text('never loaded')
    return dict(fits=[dict(fit_id='first', input_file='window.npz', constants_file='constants.json',
                          gparam_file='sensor.dat', raw_fs=100., config={'niter': 200, 'burnin': 100})]), binary


def test_preparation_freezes_defaults_settings_seeds_and_environment(spec, tmp_path):
    value, binary = spec
    prepared = prepare_manifest(value, binary, workers=2, mps='require', cpu_placement='separate', base=tmp_path)
    fit = prepared['fits'][0]
    assert fit['n_frames'] == 5 and fit['config']['noise_calibration_method'] == 'diff'
    assert fit['seeded_constants']['MCMC']['seed'] == fit['seed'] == pool.stable_seed('first')
    assert prepared['execution']['mps'] == 'require' and prepared['execution']['workers'] == 2
    assert prepared['python_environment'] == pool.python_environment()
    assert 'pgas_bound_gpu' not in sys.modules
    assert value['fits'][0]['config'] == {'niter': 200, 'burnin': 100}
    # Relocating the launcher must still pin the whole package, including CLI
    # and inference defaults outside the pgas subpackage.
    package = Path(pool.__file__).resolve().parents[1]
    for relative in ('pgas/pool.py', 'cli/pgas_pool.py', 'cli/slurm_mps.py', 'inference/pgas.py'):
        path = package / relative
        assert prepared['runtime_files_sha256'][str(path)] == pool.sha256(path)
    pool.validate_manifest(prepared)
    changed = copy.deepcopy(prepared); changed['python_environment']['numpy'] = 'different'
    with pytest.raises(ValueError, match='environment changed'): pool.validate_manifest(changed)


@pytest.mark.parametrize('variable,value', [
    ('C_SPIKES_PGAS_RESAMPLING', 'device'),
    ('C_SPIKES_PGAS_TRAJECTORY', 'selected'),
    ('C_SPIKES_PGAS_ANCESTOR_SEED', '123'),
])
def test_changed_native_options_rejected_before_launch(spec, tmp_path, monkeypatch, variable, value):
    source, binary = spec
    for name in ('C_SPIKES_PGAS_RESAMPLING', 'C_SPIKES_PGAS_TRAJECTORY', 'C_SPIKES_PGAS_ANCESTOR_SEED'):
        monkeypatch.delenv(name, raising=False)
    prepared = prepare_manifest(source, binary, base=tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', 'fake-test-job')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'MIG-test-only')
    monkeypatch.setenv(variable, value)
    root = tmp_path / 'out'
    with pytest.raises(ValueError, match='execution modes differ'):
        pool.run_manifest(prepared, root, resume=True)
    assert not root.exists()


@pytest.mark.parametrize('override', [{'mystery_mode': True}, {'use_cache': True},
                                     {'edges': [[0, .02]]}, {'bm_sigma_use_low_activity_mask': True}])
def test_preparation_refuses_unknown_or_unsupported_modes(spec, tmp_path, override):
    value, binary = spec; value['fits'][0]['config'].update(override)
    with pytest.raises(ValueError): prepare_manifest(value, binary, base=tmp_path)


def test_cli_prepare_and_help_do_not_need_a_gpu_or_slurm(spec, tmp_path):
    value, binary = spec
    source = tmp_path/'fits.json'; source.write_text(json.dumps(value))
    target = tmp_path/'frozen.json'
    env = dict(os.environ); env.pop('SLURM_JOB_ID', None); env['CUDA_VISIBLE_DEVICES'] = ''
    argv = [sys.executable, '-m', 'c_spikes.cli.pgas_pool', '--prepare', str(source),
            '--binary', str(binary), '--manifest', str(target)]
    result = subprocess.run(argv, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    frozen = json.loads(target.read_text())
    assert frozen['execution']['workers'] == 1 and frozen['execution']['mps'] == 'off'
    assert subprocess.run(argv, env=env, capture_output=True).returncode != 0
    assert subprocess.run([sys.executable, '-m', 'c_spikes.cli.pgas_pool', '--help'], env=env,
                          capture_output=True).returncode == 0


def test_prepared_manifest_drives_python_api_without_repeating_modes(spec, tmp_path, monkeypatch):
    value, binary = spec
    monkeypatch.setenv('C_SPIKES_PGAS_RESAMPLING', 'device')
    monkeypatch.setenv('C_SPIKES_PGAS_TRAJECTORY', 'selected')
    monkeypatch.setenv('C_SPIKES_PGAS_ANCESTOR_SEED', '123')
    prepared = prepare_manifest(value, binary, workers=2, mps='off', base=tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', str(len(os.sched_getaffinity(0))))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'MIG-test-only')
    fake = Path(__file__).with_name('pgas_fake_worker.py')
    original = pool.run_one
    monkeypatch.setattr(pool, 'run_one', lambda *a, **k: original(*a, **k, command=[sys.executable, str(fake)]))
    report = pool.run_manifest(prepared, tmp_path/'out')
    assert report['workers'] == 2 and report['execution']['mps'] == 'off'
    assert report['new_successes'] == 1
    result = json.loads((Path(report['fits'][0]['output']) / 'answer.json').read_text())
    assert result['pgas_environment'] == prepared['execution']['pgas_environment']


def test_default_worker_entrypoint_preserves_failure_diagnostics(spec, tmp_path, monkeypatch):
    value, binary = spec
    prepared = prepare_manifest(value, binary, base=tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', 'fake-test-job')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'MIG-test-only')
    # Use the real relocated worker entry point. This intentionally invalid
    # binary fails before CUDA, and must still leave the worker's diagnostics.
    report = pool.run_manifest(prepared, tmp_path / 'out')
    assert report['failures'] == 1
    attempt = Path(report['fits'][0]['output'])
    error = json.loads((attempt / 'worker-error.json').read_text())
    assert error['error'].startswith('ImportError:')
    assert json.loads((attempt / 'process-start.json').read_text())['argv'][2] == 'c_spikes.cli.pgas_pool'
