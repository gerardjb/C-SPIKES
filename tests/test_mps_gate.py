"""CPU-only checks of failure evidence, service scope and simultaneous gating."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux" or sys.version_info < (3, 9),
                                reason="Launcher requires Linux and Python 3.9+")

from c_spikes.pgas import mps_gate as gate, mps


def test_scope_requires_job_uuid_and_allocated_cores(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','123')
    row=dict(uid=os.getuid(),cgroup='0::/slurm/job_123/step_batch',
             environment={'CUDA_VISIBLE_DEVICES':'MIG-assigned'},thread_affinity={'7':[2,3]})
    gate.verify_scope(row,'MIG-assigned',[2,3],'/tmp/nvidia-mps')
    for change in [dict(cgroup='0::/slurm/job_1234/step_batch'),
                   dict(environment={}),dict(thread_affinity={'7':[2,4]})]:
        with pytest.raises(RuntimeError):gate.verify_scope(dict(row,**change),'MIG-assigned',[2,3],'/tmp/nvidia-mps')


def test_exact_cgroup_device_proof_can_replace_missing_service_filter(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','123')
    row=dict(uid=os.getuid(),cgroup='0::/slurm/job_123/step_batch',environment={},thread_affinity={'7':[2,3]})
    proof=dict(verified=True,cgroup=row['cgroup'],device_uuids=['MIG-assigned'])
    gate.verify_scope(row,'MIG-assigned',[2,3],'/tmp/nvidia-mps',proof)
    for change in [dict(verified=False),dict(cgroup='other'),dict(device_uuids=['MIG-assigned','MIG-other'])]:
        with pytest.raises(RuntimeError):gate.verify_scope(row,'MIG-assigned',[2,3],'/tmp/nvidia-mps',dict(proof,**change))
    with pytest.raises(RuntimeError):
        gate.verify_scope(dict(row,environment={'CUDA_VISIBLE_DEVICES':'MIG-other'}),'MIG-assigned',[2,3],'/tmp/nvidia-mps',proof)


def test_fallback_persists_environment_driver_and_control_error(tmp_path,monkeypatch):
    monkeypatch.setattr(mps,'private_pipe',lambda:tmp_path)
    monkeypatch.setattr(mps,'cuda_mps_enabled',lambda:0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','MIG-assigned')
    monkeypatch.setenv('SLURM_JWT','must-not-be-captured')
    monkeypatch.setattr(gate.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='No MPS control daemon'))
    path=tmp_path/'failed.json'
    with pytest.raises(RuntimeError,match='fallback'):gate.diagnostic_snapshot(True,path)
    record=json.loads(path.read_text())
    assert record['driver_mps_enabled']==0 and not record['verified']
    assert record['environment']['CUDA_VISIBLE_DEVICES']=='MIG-assigned'
    assert 'SLURM_JWT' not in record['environment']
    assert record['queries'][0]['returncode']==1
    assert record['queries'][0]['stderr']=='No MPS control daemon'


def test_membership_failure_still_keeps_query_evidence(tmp_path,monkeypatch):
    (tmp_path/'nvidia-cuda-mps-control.pid').write_text('99')
    monkeypatch.setattr(mps,'private_pipe',lambda:tmp_path)
    monkeypatch.setattr(mps,'cuda_mps_enabled',lambda:1)
    monkeypatch.setattr(gate,'query',lambda *a:dict(command=a[0],returncode=2,stdout='',stderr='server unavailable'))
    def fail(ids):raise RuntimeError('No client membership')
    monkeypatch.setattr(mps,'service_snapshot',fail)
    path=tmp_path/'failed.json'
    with pytest.raises(RuntimeError,match='membership'):gate.diagnostic_snapshot(True,path)
    record=json.loads(path.read_text())
    assert record['driver_mps_enabled']==1 and not record['verified']
    assert record['queries'][0]['stderr']=='server unavailable'


def test_control_timeout_is_durable(tmp_path,monkeypatch):
    monkeypatch.setattr(mps,'private_pipe',lambda:tmp_path)
    monkeypatch.setattr(mps,'cuda_mps_enabled',lambda:0)
    def timeout(*a,**k):raise subprocess.TimeoutExpired('mps',5,output=b'partial',stderr=b'timeout detail')
    monkeypatch.setattr(gate.subprocess,'run',timeout)
    path=tmp_path/'failed.json'
    with pytest.raises(RuntimeError):gate.diagnostic_snapshot(True,path)
    row=json.loads(path.read_text())['queries'][0]
    assert row['returncode'] is None and row['stdout']=='partial' and row['stderr']=='timeout detail'


def test_unscoped_control_endpoint_is_never_contacted(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Must not query an unverified endpoint')
    monkeypatch.setattr(gate.subprocess,'run',forbidden)
    result=gate.query('get_server_list',tmp_path,999999999)
    assert result['returncode'] is None and 'error' in result


def test_inference_identity_error_keeps_environment_and_diagnostics(tmp_path,monkeypatch):
    import importlib.util
    import numpy  # CPU import before mocking the native extension loader.
    from c_spikes.pgas import pool
    cpu=min(os.sched_getaffinity(0))
    task=tmp_path/'task.json'
    task.write_text(json.dumps(dict(fit={'fit_id':'test','files_sha256':{}},
        binary='unused-native',runtime_files_sha256={},worker_cpu=cpu,allocated_cpus=[cpu],mps_expected=True)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','MIG-test')
    monkeypatch.setattr(pool,'apply_worker_affinity',lambda task:None)
    original_spec=importlib.util.spec_from_file_location
    original_module=importlib.util.module_from_spec
    dummy=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None))
    monkeypatch.setattr(importlib.util,'spec_from_file_location',lambda name,*a,**k:dummy if name=='pgas_bound_gpu' else original_spec(name,*a,**k))
    monkeypatch.setattr(importlib.util,'module_from_spec',lambda spec:SimpleNamespace() if spec is dummy else original_module(spec))
    def fail():raise RuntimeError('CUDA identity unavailable')
    monkeypatch.setattr(pool,'cuda_identity',fail)
    monkeypatch.setattr(mps,'private_pipe',lambda:tmp_path)
    monkeypatch.setattr(mps,'cuda_mps_enabled',lambda:0)
    monkeypatch.setattr(gate.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='No daemon'))
    with pytest.raises(RuntimeError,match='CUDA identity unavailable'):pool.fit_worker(task)
    start=json.loads((tmp_path/'execution-start.json').read_text())
    assert start['environment']['CUDA_VISIBLE_DEVICES']=='MIG-test'
    assert start['error']=='CUDA identity unavailable'
    assert json.loads((tmp_path/'mps-start.json').read_text())['queries'][0]['stderr']=='No daemon'


@pytest.mark.parametrize('mode',['pass','partner_failure','membership_failure'])
def test_two_real_processes_cannot_start_work_before_joint_verification(tmp_path,monkeypatch,mode):
    barrier=dict(directory=str(tmp_path),token='unique-test',participants=['one','two'],timeout_s=5)
    config=tmp_path/'barrier.json';config.write_text(json.dumps(barrier))
    child=tmp_path/'child.py';child.write_text('''
import json, os, pathlib, sys, time
from c_spikes.pgas.mps_gate import wait_for_release
barrier=json.loads(pathlib.Path(sys.argv[1]).read_text());label=sys.argv[2]
time.sleep(float(sys.argv[3]))
if sys.argv[4]=='fail': raise SystemExit(2)
wait_for_release(barrier,label,dict(pid=os.getpid(),mps={'verified':True,'driver_mps_enabled':1}))
(pathlib.Path(barrier['directory'])/(label+'.expensive-work')).write_text('started')
''')
    clients=[subprocess.Popen([sys.executable,str(child),str(config),'one','0','ok'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable,str(child),str(config),'two','.3','fail' if mode=='partner_failure' else 'ok'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)]
    def verify(ids):
        assert set(ids)=={p.pid for p in clients}
        assert all(p.poll() is None for p in clients)
        assert not list(tmp_path.glob('*.expensive-work'))
        if mode=='membership_failure':raise RuntimeError('Second client absent from service')
        return dict(required_clients=ids,server_pid=99)
    monkeypatch.setattr(mps,'service_snapshot',verify)
    try:
        decision=gate.release_when_verified(barrier,lambda:any(p.poll() is not None for p in clients))
        for p in clients:p.wait(timeout=5)
        assert decision['released']==(mode=='pass')
        assert len(list(tmp_path.glob('*.expensive-work')))==(2 if mode=='pass' else 0)
    finally:
        for p in clients:
            if p.poll() is None:p.kill();p.wait()
