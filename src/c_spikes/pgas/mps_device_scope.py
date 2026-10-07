"""Read-only CUDA enumeration in a job cgroup, without an MPS endpoint/context."""
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import uuid


def main():
    record = dict(pid=os.getpid(), cgroup=Path('/proc/self/cgroup').read_text(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  pipe_directory=os.environ['CUDA_MPS_PIPE_DIRECTORY'], calls=[], devices=[])
    try:
        if record['cuda_visible_devices'] is not None or any(Path(record['pipe_directory']).iterdir()):
            raise RuntimeError('Enumeration requires unset device filter and a fresh empty private pipe')
        driver=ct.CDLL('libcuda.so.1')
        def call(name,args,types):
            fn=getattr(driver,name);fn.argtypes=types;fn.restype=ct.c_int
            rc=fn(*args);record['calls'].append(dict(function=name,returncode=rc))
            if rc:raise RuntimeError(f'{name} returned {rc}')
        call('cuInit',[0],[ct.c_uint])
        version=ct.c_int();count=ct.c_int()
        call('cuDriverGetVersion',[ct.byref(version)],[ct.POINTER(ct.c_int)])
        record['driver_api_version']=version.value
        call('cuDeviceGetCount',[ct.byref(count)],[ct.POINTER(ct.c_int)])
        for ordinal in range(count.value):
            device=ct.c_int();uid=(ct.c_ubyte*16)();name=ct.create_string_buffer(128)
            call('cuDeviceGet',[ct.byref(device),ordinal],[ct.POINTER(ct.c_int),ct.c_int])
            call('cuDeviceGetUuid_v2',[ct.byref(uid),device],[ct.c_void_p,ct.c_int])
            call('cuDeviceGetName',[name,len(name),device],[ct.c_char_p,ct.c_int,ct.c_int])
            label = name.value.decode()
            prefix = 'MIG-' if 'MIG ' in label else 'GPU-'
            record['devices'].append(dict(uuid=prefix+str(uuid.UUID(bytes=bytes(uid))),name=label))
        record['completed']=True
    except Exception as exc:record.update(completed=False,error=str(exc))
    Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
    return 0 if record['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
