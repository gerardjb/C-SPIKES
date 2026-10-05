"""Exercise real child processes, isolation, receipts, and recovery without CUDA."""
import json
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux" or sys.version_info < (3, 9),
                                reason="Launcher requires Linux and Python 3.9+")

from c_spikes import pgas_pool as pool


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.write_text("pinned input")
    fake = tmp_path / "fake.py"
    fake.write_text('''
import json, os, pathlib, time
from c_spikes.pgas_pool import atomic_json, task_identity, sha256, apply_worker_affinity
out=pathlib.Path.cwd()
task=json.loads((out/'task.json').read_text())
inherited_affinity=sorted(os.sched_getaffinity(0))
apply_worker_affinity(task)
fit=task['fit']
if fit.get('fail'): raise SystemExit(9)
time.sleep(fit.get('delay', 0))
atomic_json(out/'answer.json', {'id':fit['fit_id'], 'seed':fit['seed'],
                              'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
                              'threads':os.environ['OPENBLAS_NUM_THREADS']})
atomic_json(out/'completion.json', {'task_sha256':task_identity(task),
             'execution':task['execution'], 'seed':fit['seed'],
             'inherited_affinity':inherited_affinity,
             'cpu_affinity':sorted(os.sched_getaffinity(0)),
             'artifacts_sha256':{'answer.json':sha256(out/'answer.json')}})
''')
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(len(os.sched_getaffinity(0))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-test-only")
    original = pool.run_one
    def run_fake(*args, **kwargs):
        return original(*args, **kwargs, command=[sys.executable, str(fake)])
    monkeypatch.setattr(pool, "run_one", run_fake)
    def fit(name, **extra):
        return dict(fit_id=name, seed=pool.stable_seed(name), config=dict(burnin=100, niter=200),
                    files_sha256={str(source):pool.sha256(source)}, input_file=str(source),
                    constants_file=str(source), gparam_file=str(source), **extra)
    manifest = dict(fits=[fit("slow", delay=.15), fit("fast")], binary=str(source),
                    runtime_files_sha256={str(source):pool.sha256(source)})
    return manifest, tmp_path / "output", fit


def test_fresh_process_order_identity_and_inherited_device(campaign):
    manifest, root, _ = campaign
    report = pool.run_manifest(manifest, root, workers=2)
    assert [r["fit_id"] for r in report["fits"]] == ["slow", "fast"]
    assert report["new_successes"] == 2
    for row in report["fits"]:
        answer = json.loads((Path(row["output"]) / "answer.json").read_text())
        assert answer == dict(id=row["fit_id"], seed=pool.stable_seed(row["fit_id"]),
                              gpu="MIG-test-only", threads="1")
    reordered = dict(manifest, fits=list(reversed(manifest["fits"])))
    other = pool.run_manifest(reordered, root.parent / "reordered", workers=1)
    assert {r["fit_id"]:r["receipt"]["artifacts_sha256"] for r in report["fits"]} == {
        r["fit_id"]:r["receipt"]["artifacts_sha256"] for r in other["fits"]}


def test_duplicate_and_collision_rejected_before_start(campaign):
    manifest, root, _ = campaign
    with pytest.raises(ValueError, match="unique"):
        pool.run_manifest(dict(manifest, fits=manifest["fits"] * 2), root)
    assert not root.exists()
    pool.run_manifest(manifest, root)
    with pytest.raises(FileExistsError):
        pool.run_manifest(manifest, root)
    manifest["fits"][0]["config"]["niter"] = 400
    with pytest.raises(ValueError, match="another manifest"):
        pool.run_manifest(manifest, root, resume=True)


def test_failure_isolation_partial_recovery_and_cached_count(campaign):
    manifest, root, fit = campaign
    manifest["fits"].append(fit("failed", fail=True))
    report = pool.run_manifest(manifest, root, workers=2)
    assert [r["status"] for r in report["fits"]] == ["completed", "completed", "failed"]
    assert (root / "failed/attempt-0001/failure.json").is_file()
    with pytest.raises(FileExistsError, match="Partial"):
        pool.run_manifest(manifest, root, workers=2, resume=True)
    retry = pool.run_manifest(manifest, root, workers=2, resume=True, retry_failed=True)
    assert retry["new_successes"] == 0
    assert [r["status"] for r in retry["fits"]] == ["cached", "cached", "failed"]
    assert (root / "failed/attempt-0002/failure.json").is_file()


def test_resume_rechecks_artifacts(campaign):
    manifest, root, _ = campaign
    pool.run_manifest(manifest, root)
    report = pool.run_manifest(manifest, root, resume=True)
    assert report["new_successes"] == 0
    (root / "slow/attempt-0001/answer.json").write_text("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        pool.run_manifest(manifest, root, resume=True)


def test_resource_and_seed_guards(campaign, monkeypatch):
    manifest, root, _ = campaign
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="Exactly one"):
        pool.run_manifest(manifest, root, workers=2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(RuntimeError, match="Slurm"):
        pool.run_manifest(manifest, root)
    manifest["fits"][0]["seed"] += 1
    with pytest.raises(ValueError, match="Seed"):
        pool.validate_manifest(manifest)


def test_concurrent_coordinators_cannot_share_output(campaign):
    import fcntl
    manifest, root, _ = campaign
    root.mkdir()
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            pool.run_manifest(manifest, root)


def test_real_workers_same_and_separate_core_with_fixed_coordinator(campaign):
    manifest, root, _ = campaign
    original = os.sched_getaffinity(0)
    cores = {}
    for cpu in sorted(original):
        detail = pool.cpu_identity(cpu)
        cores.setdefault((detail['socket'], detail['core']), cpu)
    if len(cores) < 2:
        pytest.skip('Requires a scheduled CPU allocation with two physical cores')
    first, second = list(cores.values())[:2]
    ids = [fit['fit_id'] for fit in manifest['fits']]
    for label, targets in [('B', [first, first]), ('C', [first, second])]:
        report = pool.run_manifest(manifest, root/label, workers=2,
                    worker_cpus=dict(zip(ids,targets)), coordinator_cpu=first)
        assert [row['receipt']['cpu_affinity'] for row in report['fits']] == [[cpu] for cpu in targets]
        assert all(row['receipt']['inherited_affinity']==[first] for row in report['fits'])
        assert os.sched_getaffinity(0) == original
    # Reusing B's output with C's execution identity must be refused.
    with pytest.raises(ValueError, match='another manifest'):
        pool.run_manifest(manifest, root/'B', workers=2, resume=True,
                          worker_cpus=dict(zip(ids,[first,second])), coordinator_cpu=first)


def test_cpu_plan_rejects_unallocated_or_missing_cpu_before_work(campaign):
    manifest, root, _ = campaign
    cpus = os.sched_getaffinity(0)
    plan = {fit['fit_id']:min(cpus) for fit in manifest['fits']}
    plan[manifest['fits'][0]['fit_id']] = max(cpus)+10000
    with pytest.raises(ValueError, match='inside the allocation'):
        pool.run_manifest(manifest, root, workers=2, worker_cpus=plan, coordinator_cpu=min(cpus))
    assert not root.exists()
    with pytest.raises(ValueError, match='inside the allocation'):
        pool.run_manifest(manifest, root, workers=2, worker_cpus={}, coordinator_cpu=min(cpus))
