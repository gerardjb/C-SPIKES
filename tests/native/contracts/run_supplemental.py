"""Run the retained self-asserting checked fixtures in fresh CPU processes.

This does not supply historical numerical/RNG parity oracles. Build and admit the
explicit checked extension/targets before invocation; no builds or fallback occur.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=float, required=True)
    parser.add_argument('--include-core-guards', action='store_true',
                        help='Also invoke existing checked math and transaction cases (for OFF validation)')
    args = parser.parse_args()
    if sys.flags.optimize or args.deadline_seconds <= 0:
        raise RuntimeError('Assertions enabled and a positive deadline are required')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError('Require one-thread environment: ' + key)
    root = Path(__file__).resolve().parent
    gp, cp, sweep = (root / 'fixtures' / name for name in ('gparams.dat', 'cparams.dat', 'sweep.json'))
    build = args.build_dir.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    python = args.python.resolve(strict=True)
    cases = []
    for case in ['C01', 'C02'] + [f'P{i:02}' for i in range(1, 5)]:
        cases.append(('factory-' + case, [build / 'csp_contract_factory', case, gp, cp]))
    for target, prefix in [('sweep_preparation', 'T'), ('sweep_run', 'F')]:
        for case in [f'{prefix}{i:02}' for i in range(1, 9)] + [f'P{i:02}' for i in range(1, 9)]:
            cases.append((target + '-' + case, [build / ('csp_contract_' + target), case, gp, sweep]))
    for case in ['T12', 'T13', 'T14']:
        cases.append(('python-' + case, [python, root / 'checked_python_case.py', extension, case, gp, cp]))
    for case in ['C03', 'C04']:
        cases.append(('python-' + case, [python, root / 'factory_python_case.py', case, extension.parent, gp, cp]))
    if args.include_core_guards:
        cases.append(('checked-math', [build / 'csp_contract_checked_math']))
        for case in [f'T{i:02}' for i in range(1, 12)] + ['T15', 'T16']:
            cases.append(('transaction-' + case, [build / 'csp_contract_checked_transactions', case, gp, cp]))
    inputs = {Path(cmd[0]) for _, cmd in cases} | {extension, gp, cp, sweep, root / 'checked_python_case.py', root / 'factory_python_case.py'}
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    args.output.mkdir(parents=True, exist_ok=False)
    ledger = []
    deadline = time.monotonic() + args.deadline_seconds
    def save():
        (args.output / 'LEDGER.json').write_text(json.dumps({'inputs': hashes, 'cases': ledger}, indent=2) + '\n')
    for name, command in cases:
        remaining = min(30, deadline - time.monotonic() - 5)
        if remaining <= 0:
            raise RuntimeError('No remaining deadline/cleanup allowance')
        cwd = args.output / name
        cwd.mkdir()
        row = {'id': name, 'argv': list(map(str, command)), 'status': 'started'}
        ledger.append(row)
        save()
        child = None
        started = time.monotonic()
        try:
            with (cwd / 'stdout').open('wb') as stdout, (cwd / 'stderr').open('wb') as stderr:
                child = subprocess.Popen(row['argv'], cwd=cwd, stdout=stdout, stderr=stderr,
                                         start_new_session=True, stdin=subprocess.DEVNULL)
                row['returncode'] = child.wait(timeout=remaining)
            if row['returncode'] != 0:
                raise RuntimeError(name + ': fixture assertion/process failed')
            unexpected = sorted(p.name for p in cwd.iterdir() if p.name not in {'stdout', 'stderr'})
            if unexpected:
                raise RuntimeError(name + ': unexpected output files: ' + repr(unexpected))
            row['status'] = 'passed_local_assertions'
        except BaseException as exc:
            row.update(status='failed', error=repr(exc))
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
            row['elapsed_seconds'] = time.monotonic() - started
            save()
    (args.output / 'SUMMARY.json').write_text(json.dumps({'passed': len(ledger), 'scope': 'local self-assertions; no independent P-case numerical/RNG parity'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
