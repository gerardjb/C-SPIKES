"""Real subprocess queue/recovery tests with simulated CUDA attachment evidence."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux" or sys.version_info < (3, 9),
                                reason="Launcher requires Linux and Python 3.9+")

from c_spikes import pgas_pool as pool, pgas_queue as queue, mps_gate as gate

FAKE = Path(__file__).with_name('pgas_fake_worker.py')


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    data = tmp_path/'input'; data.write_text('pinned data')
    monkeypatch.setenv('SLURM_JOB_ID', 'fake-test-job')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', str(len(os.sched_getaffinity(0))))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'MIG-test-only')
    original = pool.run_one
    monkeypatch.setattr(pool, 'run_one', lambda *a, **k: original(*a, **k, command=[sys.executable, str(FAKE)]))
    def fit(name, **kw):
        return dict(fit_id=name, seed=pool.stable_seed(name), config=dict(burnin=100, niter=200),
                    files_sha256={str(data): pool.sha256(data)}, input_file=str(data),
                    constants_file=str(data), gparam_file=str(data), **kw)
    manifest = dict(fits=[fit('slow', delay=.8), fit('fast'), fit('next'), fit('last')],
                    binary=str(data), runtime_files_sha256={str(data): pool.sha256(data)})
    return manifest, tmp_path/'out', fit


def answer(row):
    return json.loads((Path(row['output'])/'answer.json').read_text())


def fake_service(tmp_path, monkeypatch):
    path = tmp_path/'service.json'; path.write_text(json.dumps({'pipe_directory': '/fake/never-contacted'}))
    calls = []
    def inspect(ids, env, target):
        calls.append(ids)
        assert all(Path(f'/proc/{pid}').exists() for pid in ids)
        assert env['CUDA_VISIBLE_DEVICES'] == 'MIG-test-only'
        return dict(required_clients=ids, server_pid=123)
    monkeypatch.setattr(gate, 'snapshot_in_environment', inspect)
    return path, calls


@pytest.mark.parametrize('mps', ['off', 'require'])
def test_uneven_queue_refills_free_core_and_keeps_result_order(jobs, tmp_path, monkeypatch, mps):
    manifest, root, _ = jobs
    try:
        cpus = queue.cpu_slots(2, 'separate', sorted(os.sched_getaffinity(0)))
    except ValueError:
        pytest.skip('Two allocated physical cores required')
    service, calls = fake_service(tmp_path, monkeypatch)
    report = pool.run_manifest(manifest, root, workers=2, mps_mode=mps,
        cpu_placement='separate', service_file=service)
    assert [r['fit_id'] for r in report['fits']] == ['slow', 'fast', 'next', 'last']
    assert report['new_successes'] == 4 and not report['failures']
    results = [answer(row) for row in report['fits']]
    assert results[2]['begin'] < results[0]['end']  # No wave-sized tail stall.
    assert [r['affinity'] for r in results] == [[cpus[0]], [cpus[1]], [cpus[1]], [cpus[1]]]
    if mps == 'require':
        assert [len(ids) for ids in calls] == [2, 1, 1]
        assert {pid for ids in calls for pid in ids} == {r['pid'] for r in results}
    old_calls = len(calls)
    resumed = pool.run_manifest(manifest, root, workers=2, mps_mode=mps,
        cpu_placement='separate', service_file=None, resume=True)
    assert resumed['cached'] == 4 and resumed['new_successes'] == 0
    assert len(calls) == old_calls  # Cached work doesn't attach or launch.


@pytest.mark.parametrize('field', ['settings', 'input', 'runtime', 'mode', 'seed', 'placement', 'workers'])
def test_changed_identity_rejected_before_start(jobs, field):
    manifest, root, _ = jobs
    pool.run_manifest(manifest, root, workers=2, mps_mode='off', cpu_placement='shared')
    manifest = copy.deepcopy(manifest)
    opts = dict(workers=2, mps_mode='off', cpu_placement='shared')
    if field == 'settings': manifest['fits'][-1]['config']['noise_calibration_method'] = 'psd'
    if field == 'input': Path(manifest['fits'][-1]['input_file']).write_text('changed')
    if field == 'runtime': manifest['runtime_files_sha256'][manifest['binary']] = 'bad'
    if field == 'mode': opts['mps_mode'] = 'require'
    if field == 'seed':
        manifest['fits'][-1]['replicate'] = 1
        manifest['fits'][-1]['seed'] = pool.stable_seed('last', 1)
    if field == 'placement': opts['cpu_placement'] = 'separate'
    if field == 'workers': opts['workers'] = 1
    with pytest.raises(ValueError): pool.run_manifest(manifest, root, resume=True, **opts)
    assert not list(root.glob('*/attempt-0002'))


@pytest.mark.parametrize('failure', ['fail_first', 'fail_after_receipt'])
def test_failed_workers_are_isolated_and_receipt_without_success_is_not_reused(jobs, failure):
    manifest, root, fit = jobs
    manifest['fits'] = [fit('broken', **{failure: True}), fit('good'), fit('queued')]
    first = pool.run_manifest(manifest, root, workers=2)
    assert [r['status'] for r in first['fits']] == ['failed', 'completed', 'completed']
    with pytest.raises(FileExistsError, match='Partial'):
        pool.run_manifest(manifest, root, workers=2, resume=True)
    second = pool.run_manifest(manifest, root, workers=2, resume=True, retry_failed=True)
    assert [r['status'] for r in second['fits']] == ['completed', 'cached', 'cached']
    assert Path(second['fits'][0]['output']).name == 'attempt-0002'


@pytest.mark.parametrize('damage', ['task', 'process', 'receipt', 'artifact'])
def test_partial_attempts_preserved_and_retried_fresh(jobs, damage):
    manifest, root, fit = jobs; manifest['fits'] = [fit('one')]
    first = pool.run_manifest(manifest, root)
    out = Path(first['fits'][0]['output'])
    name = {'task': 'task.json', 'process': 'process.json', 'receipt': 'completion.json', 'artifact': 'answer.json'}[damage]
    (out/name).write_text('{partial')
    with pytest.raises((ValueError, FileExistsError)):
        pool.run_manifest(manifest, root, resume=True)
    second = pool.run_manifest(manifest, root, resume=True, retry_failed=True)
    assert second['new_successes'] == 1
    assert (out/name).read_text() == '{partial'


def test_cancellation_terminates_owned_workers_and_can_resume(jobs):
    manifest, root, fit = jobs
    manifest['fits'] = [fit('a', delay=.5), fit('b', delay=.5), fit('c')]
    stop = threading.Event()
    timer = threading.Timer(.2, stop.set); timer.start()
    first = pool.run_manifest(manifest, root, workers=2, stop_event=stop)
    timer.join()
    assert first['cancelled'] and first['cancelled_fits'] == 3
    assert not (root/'c').exists()
    second = pool.run_manifest(manifest, root, workers=2, resume=True, retry_failed=True)
    assert second['new_successes'] == 3 and not second['cancelled']


def test_mps_resume_uses_new_tokens_and_does_not_gate_cached_workers(jobs, tmp_path, monkeypatch):
    manifest, root, fit = jobs
    manifest['fits'] = [fit('bad', fail_first=True), fit('good')]
    service, calls = fake_service(tmp_path, monkeypatch)
    opts = dict(workers=2, mps_mode='require', cpu_placement='shared', service_file=service)
    first = pool.run_manifest(manifest, root, **opts)
    second = pool.run_manifest(manifest, root, resume=True, retry_failed=True, **opts)
    assert [len(c) for c in calls] == [2, 1]
    assert second['new_successes'] == 1 and second['cached'] == 1
    assert first['startup_gates'][0]['token'] != second['startup_gates'][0]['token']


def test_failed_initial_mps_partner_prevents_all_inference(jobs, tmp_path, monkeypatch):
    manifest, root, fit = jobs
    manifest['fits'] = [fit('bad', fail_before_gate=True), fit('good'), fit('pending')]
    service, calls = fake_service(tmp_path, monkeypatch)
    result = pool.run_manifest(manifest, root, workers=2, mps_mode='require',
                              cpu_placement='shared', service_file=service, startup_timeout=2)
    assert result['cancelled'] and not calls
    assert not list(root.glob('*/attempt-*/began.json'))
    assert (root/'bad/attempt-0001/fake-diagnostic.json').exists()


def test_startup_timeout_keeps_decision_and_rejects_work(tmp_path, monkeypatch):
    barrier = dict(directory=str(tmp_path), token='unique', participants=['absent'], timeout_s=.01)
    result = gate.release_when_verified(barrier, lambda: False)
    assert not result['released'] and 'deadline' in result['error']
    assert json.loads((tmp_path/'decision.json').read_text()) == result


def test_wrong_ready_pid_cannot_release_gate(tmp_path):
    barrier = dict(directory=str(tmp_path), token='unique', participants=['fit'], timeout_s=1)
    pool.atomic_json(tmp_path/'fit.json', dict(label='fit', pid=123, token='unique',
                                            mps=dict(verified=True, driver_mps_enabled=1)))
    result = gate.release_when_verified(barrier, lambda: False, expected_pids=lambda: {'fit': 456})
    assert not result['released'] and 'PID' in result['error']


def test_slot_validation_rejects_smt_siblings(monkeypatch):
    monkeypatch.setattr(pool, 'cpu_identity', lambda cpu: dict(socket=0, core=0))
    with pytest.raises(ValueError, match='physical core'):
        queue.cpu_slots(2, 'separate', [0, 1], [0, 1])


def test_signal_cancellation_cleans_only_owned_groups_and_cli_resume(jobs, tmp_path):
    manifest, root, fit = jobs
    manifest['fits'] = [fit('done'), fit('long', delay=30, grandchild=True), fit('queued', delay=30)]
    source = tmp_path/'manifest.json'; source.write_text(json.dumps(manifest))
    # Test-only adapter: the installed CLI still owns parsing, scheduling and signals.
    driver = tmp_path/'cli.py'
    driver.write_text('''
import sys
from c_spikes import pgas_pool as p
original=p.run_one
fake=sys.argv.pop(1)
p.run_one=lambda *a,**k: original(*a,**k,command=[sys.executable,fake])
p.main()
''')
    argv = [sys.executable, str(driver), str(FAKE), '--manifest', str(source),
            '--output', str(root), '--workers', '2']
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(40)'])
    log = (tmp_path/'cli.log').open('w')
    child = subprocess.Popen(argv, stdout=log, stderr=log)
    try:
        grandchild_path = root/'long/attempt-0001/grandchild.json'
        success_path = root/'done/attempt-0001/process.json'
        deadline = time.monotonic()+10
        while not (grandchild_path.exists() and success_path.exists()):
            assert child.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        grandchild = json.loads(grandchild_path.read_text())['pid']
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=8) == 130
        report = json.loads((root/'batch.json').read_text())
        assert report['cancelled'] and report['new_successes'] >= 1
        proc = Path(f'/proc/{grandchild}/stat')
        assert not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
        assert sentinel.poll() is None
        # Run with a pre-set cancellation event to preflight cached/partial state cheaply.
        stop = threading.Event(); stop.set()
        resumed = pool.run_manifest(manifest, root, workers=2, mps_mode='off',
            cpu_placement='shared', resume=True, retry_failed=True, stop_event=stop)
        assert resumed['fits'][0]['status'] == 'cached'
    finally:
        if child.poll() is None: child.kill(); child.wait()
        sentinel.terminate(); sentinel.wait()
        log.close()


def test_replacement_gate_failure_keeps_finished_results_and_stops_new_work(jobs, tmp_path, monkeypatch):
    manifest, root, fit = jobs
    manifest['fits'] = [fit('slow', delay=2), fit('fast'), fit('replacement'), fit('pending')]
    service, _ = fake_service(tmp_path, monkeypatch)
    count = 0
    def inspect(ids, env, target):
        nonlocal count
        count += 1
        if count > 1: raise RuntimeError('MPS service disappeared before replacement')
        return {'required_clients': ids, 'server_pid': 123}
    monkeypatch.setattr(gate, 'snapshot_in_environment', inspect)
    result = pool.run_manifest(manifest, root, workers=2, mps_mode='require',
        cpu_placement='shared', service_file=service)
    assert result['cancelled'] and result['fits'][1]['status'] == 'completed'
    assert not list((root/'replacement').glob('*/began.json'))
    assert not (root/'pending').exists()


def test_reused_slots_do_not_change_stable_task_identity(jobs):
    manifest, _, _ = jobs
    task = dict(schema_version=2, fit=manifest['fits'][0], binary=manifest['binary'],
                runtime_files_sha256=manifest['runtime_files_sha256'],
                execution=queue.execution_policy(2, 'require', 'separate'))
    first = dict(task, worker_cpu=2, allocated_cpus=[2, 5], mps_barrier={'token': 'old'})
    next_job = dict(task, worker_cpu=10, allocated_cpus=[10, 12], mps_barrier={'token': 'new'})
    assert pool.task_identity(first) == pool.task_identity(next_job)


def test_forged_completion_mode_is_not_reused(jobs):
    manifest, root, fit = jobs; manifest['fits'] = [fit('one')]
    report = pool.run_manifest(manifest, root, mps_mode='off', cpu_placement='shared')
    path = Path(report['fits'][0]['output'])/'completion.json'
    value = json.loads(path.read_text()); value['mps_end']['driver_mps_enabled'] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='verified MPS'):
        pool.run_manifest(manifest, root, resume=True, mps_mode='off', cpu_placement='shared')


def test_coordinator_sigkill_keeps_output_locked_until_owned_workers_exit(jobs, tmp_path):
    import fcntl
    manifest, root, fit = jobs
    manifest['fits'] = [fit('child', delay=30)]
    source = tmp_path/'manifest.json'; source.write_text(json.dumps(manifest))
    driver = tmp_path/'coordinator.py'
    driver.write_text('''
import json, pathlib, sys
from c_spikes import pgas_pool as p
original=p.run_one
fake=sys.argv[1]
p.run_one=lambda *a,**k: original(*a,**k,command=[sys.executable,fake])
p.run_manifest(json.loads(pathlib.Path(sys.argv[2]).read_text()),pathlib.Path(sys.argv[3]))
''')
    child = subprocess.Popen([sys.executable, str(driver), str(FAKE), str(source), str(root)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker = None
    try:
        started = root/'child/attempt-0001/began.json'
        deadline = time.monotonic()+10
        while not started.exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(.02)
        worker = json.loads(started.read_text())['pid']
        child.kill(); child.wait()
        with (root/'.lock').open('a') as lock:
            with pytest.raises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root/'child/attempt-0001/process.json').exists()
    finally:
        if child.poll() is None: child.kill(); child.wait()
        if worker is not None:
            try: os.killpg(worker, signal.SIGKILL)
            except ProcessLookupError: pass
