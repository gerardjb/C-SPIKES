"""Prepare, run and resume independent PGAS fits on one allocated GPU.

Each fit runs in a fresh interpreter. Defaults: one worker and MPS off.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from c_spikes.pgas.pool import atomic_json, fit_worker, run_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, help="Bounded process count; default 1 or frozen manifest value")
    parser.add_argument("--mps", choices=('off', 'require'), help="Default off; require rejects CUDA fallback")
    parser.add_argument("--cpu-placement", choices=('shared', 'separate'))
    parser.add_argument("--worker-cpus", help="Comma-separated allocated logical CPU IDs, one per reusable slot")
    parser.add_argument("--coordinator-cpu", type=int)
    parser.add_argument("--mps-service-file", type=Path, help="Verified current-allocation scope from site setup")
    parser.add_argument("--prepare", type=Path, metavar='SPEC', help="Freeze input spec to --manifest without CUDA")
    parser.add_argument("--binary", type=Path, help="Native GPU extension for --prepare")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fit", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fit:
        fit_worker(args.fit)
    elif args.prepare:
        if not args.binary or not args.manifest:
            parser.error('--prepare requires --binary and --manifest')
        if args.manifest.exists():
            parser.error('Refusing to replace an existing frozen manifest')
        from c_spikes.pgas.manifest import prepare_manifest
        manifest = prepare_manifest(json.loads(args.prepare.read_text()), args.binary,
            workers=args.workers if args.workers is not None else 1, mps=args.mps or 'off',
            cpu_placement=args.cpu_placement or 'shared', base=args.prepare.resolve().parent)
        atomic_json(args.manifest, manifest)
    else:
        if not args.manifest or not args.output:
            parser.error("--manifest and --output are required")
        manifest = json.loads(args.manifest.read_text())
        policy = manifest.get('execution', {})
        try:
            report = run_manifest(manifest, args.output,
                workers=args.workers if args.workers is not None else policy.get('workers', 1),
                resume=args.resume, retry_failed=args.retry_failed,
                mps_mode=args.mps or policy.get('mps', 'off'),
                cpu_placement=args.cpu_placement or policy.get('cpu_placement', 'shared'),
                slot_cpus=[int(c) for c in args.worker_cpus.split(',')] if args.worker_cpus else None,
                coordinator_cpu=args.coordinator_cpu, service_file=args.mps_service_file)
        except Exception as exc:
            from c_spikes.pgas.mps_gate import environment
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output/f'launcher-error-{os.getpid()}-{time.time_ns()}.json',
                            dict(error=f'{type(exc).__name__}: {exc}', environment=environment()))
            except OSError:
                pass  # stderr still reports failures when the output is not writable.
            parser.exit(1, f'{type(exc).__name__}: {exc}\n')
        print(json.dumps({k: v for k, v in report.items() if k != "fits"}, indent=2))
        if report['cancelled']:
            sys.exit(130)
        if report["failures"] or report['error']:
            sys.exit(1)


if __name__ == "__main__":
    main()
