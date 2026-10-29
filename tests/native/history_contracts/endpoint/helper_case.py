import sys,pathlib,json,importlib.util,os,random,pickle
import numpy as np
r=pathlib.Path(sys.argv[1]);id=sys.argv[2];variant=sys.argv[3]
sys.path.insert(0,str(r/'source/src'))
from c_spikes.syn_gen import simulate_state_history
plan=json.loads((r/'PLAN.json').read_text());fixture=next(x for x in plan['cases'] if x['id']==id)
ref=json.loads((r/'REFERENCE_BINARIES.json').read_text());extension=next(iter(ref.values()))['path'] if variant=='baseline' else str(next((r/'build/candidate').glob('pgas_bound_cpu*.so')))
spec=importlib.util.spec_from_file_location('pgas_bound_cpu',extension);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
gp=np.loadtxt(r/'gparams.dat');cp=np.loadtxt(r/'cparams.dat');times=np.array([float.fromhex(x) for x in fixture['time_hex']]);spikes=np.array([float.fromhex(x) for x in fixture['spikes_hex']]);order=tuple(['G'+str(i) for i in range(9)]+['BCa','Ca','Ca_in'])
def packed(a):return {'shape':list(a.shape),'dtype':a.dtype.str,'bytes':a.tobytes().hex()}
# No model work before READY/GO.
os.write(int(sys.argv[4]),b'READY\n');os.close(int(sys.argv[4]));assert sys.stdin.readline()=='GO\n'
m=native.GCaMP(gp,cp);m.setTimeStepMode(native.TimeStepMode.FIXEDLA);initial=m.steady_state(5e-8).copy()
files={str(x):x.stat().st_size for x in pathlib.Path.cwd().rglob('*') if x.is_file()};nr=pickle.dumps(np.random.get_state());pr=pickle.dumps(random.getstate());bt=times.tobytes();bs=spikes.tobytes()
h=simulate_state_history(m,times,spikes,reset=True)
assert h.state_order==order and h.buffer_approximation=='linear' and h.integration_method=='integrateOverTime' and h.internal_dt_s==1e-4
assert h.time_s.flags.owndata and h.state_M.flags.owndata and h.state_M.flags.c_contiguous
assert h.state_M.shape==(4,12) and h.time_s.shape==(4,) and h.state_M.dtype==h.time_s.dtype==np.dtype('float64')
assert times.tobytes()==bt and spikes.tobytes()==bs and not np.shares_memory(h.time_s,times) and not np.shares_memory(h.state_M,h.time_s)
assert np.array_equal(h.state_M[0],initial,equal_nan=True)
d=m.getStates();state=np.column_stack([d['G'],d['BCa'],d['Ca'],d['Ca_in']]);assert packed(state)==packed(h.state_M)
out={'time':packed(h.time_s),'state':packed(h.state_M),'order':order,'buffer':'linear','method':'integrateOverTime','dt':1e-4,'finite_counts':np.isfinite(h.state_M).sum(axis=0).tolist()}
h.state_M[:]=456;h.time_s[:]=789;d=m.getStates();assert packed(np.column_stack([d['G'],d['BCa'],d['Ca'],d['Ca_in']]))==packed(state)
assert nr==pickle.dumps(np.random.get_state()) and pr==pickle.dumps(random.getstate());assert files=={str(x):x.stat().st_size for x in pathlib.Path.cwd().rglob('*') if x.is_file()}
print('RESULT '+json.dumps(out,sort_keys=True))
