"""Test-only child: exercise the production queue and barriers without CUDA/data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from c_spikes.pgas_pool import atomic_json, apply_worker_affinity, sha256, task_identity
from c_spikes.mps_gate import wait_for_release

out = Path.cwd()
task = json.loads((out/'task.json').read_text())
fit = task['fit']
apply_worker_affinity(task)
if fit.get('fail_before_gate'):
    atomic_json(out/'fake-diagnostic.json', dict(error='simulated attachment failure'))
    raise SystemExit(7)
barrier = None
if 'mps_barrier' in task:
    barrier = wait_for_release(task['mps_barrier'], fit['fit_id'],
                     dict(pid=os.getpid(), mps={'verified': True, 'driver_mps_enabled': 1}))
begin = time.monotonic()
atomic_json(out/'began.json', dict(time=begin, pid=os.getpid()))
if fit.get('grandchild'):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(40)'])
    atomic_json(out/'grandchild.json', dict(pid=child.pid))
time.sleep(fit.get('delay', .02))
atomic_json(out/'answer.json', dict(begin=begin, end=time.monotonic(), pid=os.getpid(),
            affinity=sorted(os.sched_getaffinity(0)), seed=fit['seed']))
if fit.get('fail_first') and out.name == 'attempt-0001':
    raise SystemExit(8)
check = dict(verified=True, driver_mps_enabled=int(task['execution']['mps'] == 'require'),
             service=dict(client_list=dict(stdout=str(os.getpid())+'\n')))
atomic_json(out/'completion.json', dict(task_sha256=task_identity(task), pid=os.getpid(),
            seed=fit['seed'], execution=task['execution'],
            mps_start=check, mps_end=check, mps_barrier=barrier,
            artifacts_sha256={'answer.json': sha256(out/'answer.json')}))
if fit.get('fail_after_receipt') and out.name == 'attempt-0001':
    raise SystemExit(9)
