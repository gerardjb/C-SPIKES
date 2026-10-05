"""Prepare complete, hashed fit manifests before occupying a GPU allocation."""
from dataclasses import MISSING, fields
import json
import math
from pathlib import Path

from c_spikes.pgas_pool import sha256, stable_seed, validate_manifest, python_environment
from c_spikes.pgas_queue import execution_policy


def prepare_manifest(spec, binary, *, workers=1, mps='off', cpu_placement='shared', base=None):
    # Preparation may load CPU dependencies; the queue itself remains stdlib-only.
    import numpy as np
    from c_spikes.inference.pgas import PgasConfig
    base = Path(base or Path.cwd()).resolve()
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise ValueError('Native binary does not exist')
    if mps not in ('off', 'require') or cpu_placement not in ('shared', 'separate'):
        raise ValueError('Use mps=off/require and cpu_placement=shared/separate')
    execution = execution_policy(workers, mps, cpu_placement)
    owned = {'dataset_tag', 'output_root', 'constants_file', 'gparam_file', 'use_cache'}
    defaults = {f.name: f.default for f in fields(PgasConfig)
                if f.name not in owned and f.default is not MISSING}
    fits = []
    for item in spec['fits']:
        fit = dict(item)
        name = fit['fit_id']
        replicate = fit.setdefault('replicate', 0)
        if type(replicate) is not int or replicate < 0:
            raise ValueError('replicate must be a nonnegative integer')
        expected = stable_seed(name, replicate)
        if 'seed' in fit and fit['seed'] != expected:
            raise ValueError('Seed does not match fit ID and replicate')
        fit['seed'] = expected
        overrides = fit.get('config', {})
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown or launcher-owned PGAS settings: {sorted(unknown)}')
        fit['config'] = dict(defaults, **overrides)
        if fit['config']['edges'] is not None:
            raise ValueError('Queue entries must already be complete independent windows; edges are not supported')
        if fit['config']['bm_sigma_use_low_activity_mask']:
            raise ValueError('This window interface does not supply ground truth to noise calibration')
        fit['files_sha256'] = {}
        for key in ('input_file', 'constants_file', 'gparam_file'):
            path = Path(fit[key])
            path = (base / path).resolve() if not path.is_absolute() else path.resolve()
            fit[key] = str(path)
            fit['files_sha256'][str(path)] = sha256(path)
        with np.load(fit['input_file'], allow_pickle=False) as data:
            times, values = data['time'], data['fluorescence']
            if (times.ndim != 1 or times.shape != values.shape or len(times) < 2
                    or not np.isfinite(times).all() or not np.isfinite(values).all()
                    or not (np.diff(times) > 0).all()):
                raise ValueError('Expected finite 1-D time/fluorescence arrays with increasing time')
            if 'n_frames' in fit and fit['n_frames'] != len(times):
                raise ValueError('Declared frame count differs from input')
            fit['n_frames'] = len(times)
        if not math.isfinite(fit['raw_fs']) or fit['raw_fs'] <= 0:
            raise ValueError('raw_fs must be positive and finite')
        constants = json.loads(Path(fit['constants_file']).read_text())
        if constants['MCMC']['nparticles'] <= 0:
            raise ValueError('Particle count must be positive')
        fit['seeded_constants'] = dict(constants, MCMC=dict(constants['MCMC'], seed=expected))
        fits.append(fit)
    package = Path(__file__).resolve().parent
    # Pin the installed implementation, including config defaults, not a checkout name.
    runtime = {str(p): sha256(p) for p in package.rglob('*.py') if p.name != '_version.py'}
    runtime[str(binary)] = sha256(binary)
    manifest = dict(schema_version=2, fits=fits, binary=str(binary),
                    runtime_files_sha256=runtime, execution=execution,
                    python_environment=python_environment())
    validate_manifest(manifest)
    return manifest
