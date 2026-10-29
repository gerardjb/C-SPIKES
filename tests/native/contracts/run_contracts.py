"""Explicit opt-in native CPU contract runner.

Admission requires an independently reviewed build manifest and frozen input hash.
No build, installation, scheduler submission, fallback or oracle update is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def result(text):
    markers = list(re.finditer(r'^RESULT ', text, re.M))
    require(len(markers) == 1, 'exactly one RESULT marker required')
    payload = text[markers[0].end():].lstrip()
    value, end = json.JSONDecoder().raw_decode(payload)
    require(isinstance(value, dict), 'RESULT must be an object')
    require(not payload[end:].strip(), 'trailing RESULT content')
    return value


def score(text, spec, root):
    kind = spec['score']
    if kind == 'grouped':
        chunks = re.findall(r'^CASE_BEGIN (\w+)\n(.*?)\nCASE_END$', text, re.M | re.S)
        require([x[0] for x in chunks] == spec['cases'], 'case order/count mismatch')
        for case, body in chunks:
            p = root / 'oracles' / spec['oracle_group'] / (case + '.json')
            if p.is_file():
                require(result(body) == json.loads(p.read_text()), f'{spec["id"]}/{case}: exact JSON mismatch')
            else:
                p = p.with_suffix('.txt')
                require(body.rstrip('\n') == p.read_text().rstrip('\n'), f'{spec["id"]}/{case}: exact text mismatch')
    elif kind == 'exact_json':
        p = root / 'oracles' / spec['oracle_group'] / (spec['cases'][0] + '.json')
        require(result(text) == json.loads(p.read_text()), spec['id'] + ': exact JSON mismatch')
    elif kind == 'checked_math':
        actual = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
        require(actual == [{'case': f'F{i:02}', 'passed': True} for i in range(1, 9)], 'checked math results')
    elif kind == 'transaction':
        require(result(text) == {'id': spec['cases'][0], 'ok': True}, 'transaction result')
    elif kind == 'diagnostic':
        v = result(text)
        require(v['id'] == spec['cases'][0] and v['ok'] and set(v) == {'id', 'ok', 'captured_records'}, 'diagnostic result')
    elif kind == 'internal_assertions':
        # Collector cases P06/P07/P14/P15 assert internally; execute checks exit zero.
        pass
    else:
        raise RuntimeError('unrecognized scorer: ' + kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-manifest', type=Path, required=True)
    parser.add_argument('--expected-build-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=float, required=True)
    parser.add_argument('--run', action='store_true', help='Run only under separately authorized CPU validation')
    args = parser.parse_args()
    require(not sys.flags.optimize, 'Python optimization disables historical child assertions; do not use -O')
    require(sys.platform.startswith('linux'), 'Current probe/wrapper/provider contract is Linux only')
    require(args.deadline_seconds > 0, 'positive bounded deadline required')
    root = Path(__file__).resolve().parent
    source = root.parents[2]
    require(digest(args.build_manifest) == args.expected_build_manifest_sha256, 'build admission manifest identity mismatch')
    admission = json.loads(args.build_manifest.read_text())
    require(admission['contract_inputs_sha256'] == digest(root / 'INPUTS.json'), 'build/source contract mismatch')
    require(admission['variant'] == 'checked-cpu' and admission['observations'] == 'on', 'wrong variant or observational mode')
    require(admission['scientific_provider_policy_changed'] is False, 'provider policy change is outside curation')
    inputs = json.loads((root / 'INPUTS.json').read_text())
    for rel, expected in inputs['sha256'].items():
        require(digest(source / rel) == expected, 'input drift: ' + rel)
    specs = json.loads((root / 'cases.json').read_text())['cases']
    binaries = {}
    for target in sorted({s['target'] for s in specs}):
        item = admission['binaries'][target]
        path = Path(item['path']).resolve(strict=True)
        require(path.is_file() and os.access(path, os.X_OK), 'missing executable: ' + target)
        require(digest(path) == item['sha256'], 'binary drift: ' + target)
        binaries[target] = path
    # Every small reference is admitted before the first child, never after spending its budget.
    for spec in specs:
        if spec['score'] in ('grouped', 'exact_json'):
            for case in spec['cases']:
                files = [root / 'oracles' / spec['oracle_group'] / (case + ext) for ext in ('.json', '.txt')]
                require(sum(p.is_file() for p in files) == 1, 'missing/ambiguous reference: ' + spec['id'] + '/' + case)
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(var) == '1', 'admit one-thread environment explicitly: ' + var)
    if not args.run:
        print(json.dumps({'status': 'admitted_only', 'processes': len(specs), 'executed': False}))
        return
    # No partial-overwrite/resume by accident: each reviewed run gets a fresh output directory.
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.deadline_seconds
    os.environ['G2AD_DEADLINE'] = str(deadline)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    from processes import execute
    ledger = []
    def save():
        (args.output / 'LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n')
    try:
        for spec in specs:
            require(deadline - time.monotonic() >= 10 + 2 + spec['body_limit_seconds'], 'deadline admission leaves insufficient cleanup/report reserve')
            row = {'id': spec['id'], 'status': 'started'}
            ledger.append(row)
            save()
            cwd = args.output / (spec['id'] + '-cwd')
            cwd.mkdir()
            values = {'fixtures': str(root / 'fixtures')}
            command = [str(binaries[spec['target']])] + [a.format(**values) for a in spec['args']]
            text, timing = execute(command, cwd, args.output, spec['id'], body_limit=spec['body_limit_seconds'])
            row.update(timing)
            score(text, spec, root)
            row.update(status='passed', subcases=spec['cases'])
            save()
        for target, path in binaries.items():
            require(digest(path) == admission['binaries'][target]['sha256'], 'post-run binary drift')
        (args.output / 'SUMMARY.json').write_text(json.dumps({'passed': len(ledger), 'expected': len(specs), 'scope': 'native CPU contract subset; not complete inference or platform qualification'}, indent=2) + '\n')
    except BaseException as exc:
        if ledger and ledger[-1]['status'] == 'started':
            ledger[-1].update(status='failed', error=str(exc))
        save()
        (args.output / 'STOP.json').write_text(json.dumps({'error': str(exc), 'passed': sum(r['status'] == 'passed' for r in ledger)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
