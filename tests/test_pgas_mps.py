"""Fail-closed attachment evidence without importing or starting CUDA/MPS."""
import os
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux" or sys.version_info < (3, 9),
                                reason="Launcher requires Linux and Python 3.9+")
from c_spikes import pgas_mps as mps


def test_pid_evidence_is_not_a_substring_or_device_column():
    assert mps.pids('PID\n1234\n 2345 \nGPU 0: 12\n12 34\n')==[1234,2345]


def test_daemon_alone_and_wrong_client_do_not_establish_attachment(tmp_path,monkeypatch):
    (tmp_path/'nvidia-cuda-mps-control.pid').write_text('77')
    monkeypatch.setattr(mps,'private_pipe',lambda:tmp_path)
    def ctl(command,required=True):
        return dict(stdout='88\n' if command=='get_server_list' else '12345\n',returncode=0)
    monkeypatch.setattr(mps,'control',ctl)
    monkeypatch.setattr(mps,'process_identity',lambda pid:dict(pid=pid))
    with pytest.raises(RuntimeError,match='not attached'):mps.service_snapshot([1234])
    with pytest.raises(RuntimeError,match='not attached'):mps.service_snapshot([12345,9])
    assert mps.service_snapshot([12345])['server_pid']==88


@pytest.mark.parametrize('expected,actual',[(True,0),(False,1)])
def test_rejects_fallback_and_contaminated_off_control(tmp_path,monkeypatch,expected,actual):
    monkeypatch.setattr(mps,'private_pipe',lambda:tmp_path)
    monkeypatch.setattr(mps,'cuda_mps_enabled',lambda:actual)
    with pytest.raises(RuntimeError,match='fallback'):mps.worker_snapshot(expected)


def test_on_requires_both_driver_attribute_and_membership(tmp_path,monkeypatch):
    monkeypatch.setattr(mps,'private_pipe',lambda:tmp_path)
    monkeypatch.setattr(mps,'cuda_mps_enabled',lambda:1)
    spy=Mock(return_value={'server_pid':999})
    monkeypatch.setattr(mps,'service_snapshot',spy)
    result=mps.worker_snapshot(True)
    spy.assert_called_once_with([os.getpid()])
    assert result['driver_mps_enabled']==1 and result['service']['server_pid']==999


def test_global_or_other_job_pipe_rejected(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','123')
    for name in ('/tmp/nvidia-mps',f'/tmp/cspikes-mps-{os.getuid()}-124'):
        monkeypatch.setenv('CUDA_MPS_PIPE_DIRECTORY',name)
        with pytest.raises(RuntimeError,match='private directory'):mps.private_pipe()
