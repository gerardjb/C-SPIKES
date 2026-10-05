"""Inspect an already-running, allocation-scoped Slurm MPS service; never manage it."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for scope, control-query evidence and available logs')
    args = parser.parse_args()
    from c_spikes.pgas_pool import allocation_environment
    from c_spikes.mps_gate import discover_service
    allocation_environment(1)
    assigned = os.environ['CUDA_VISIBLE_DEVICES']
    if not assigned.startswith(('MIG-', 'GPU-')):
        parser.error('Service scope verification requires an assigned device UUID, not an ordinal')
    args.output.mkdir(parents=True, exist_ok=False)
    report = discover_service(args.output, assigned, sorted(os.sched_getaffinity(0)))
    print(json.dumps(dict(verified=report['verified'],
                         service_file=str((args.output/'service-scope.json').resolve()),
                         error=report.get('error')), indent=2))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
