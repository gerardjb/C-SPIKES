import sys,pathlib,json,importlib.util,os,random,pickle
import numpy as np
R=pathlib.Path(sys.argv[1]);id=sys.argv[2];sys.path.insert(0,str(R/'source/src'))
from c_spikes.syn_gen import simulate_state_history
extension=next((R/'build/candidate').glob('pgas_bound_cpu*.so'))
spec=importlib.util.spec_from_file_location('pgas_bound_cpu',extension);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
f=next(x for x in json.loads((R/'PLAN.json').read_text())['cases'] if x['id']==id)
gp=np.loadtxt(R/'gparams.dat');cp=np.loadtxt(R/'cparams.dat');t=np.array([float.fromhex(x) for x in f['time_hex']]);sp=np.array([float.fromhex(x) for x in f['spikes_hex']]);ca=float.fromhex(f['initial_ca_hex'])
os.write(int(sys.argv[3]),b'READY\n');os.close(int(sys.argv[3]));assert sys.stdin.readline()=='GO\n'
nr=pickle.dumps(np.random.get_state());pr=pickle.dumps(random.getstate());files={str(x):x.stat().st_size for x in pathlib.Path.cwd().rglob('*') if x.is_file()};bt=t.tobytes();bs=sp.tobytes()
m=native.GCaMP(gp,cp);m.setTimeStepMode(native.TimeStepMode.FIXEDLA);m.setState(ca);m.integrateOverTime(t,sp)
def states():
 s=m.getStates();return np.column_stack([s['G'],s['BCa'],s['Ca'],s['Ca_in']])
a=states();dff=m.getDFFValues();original=dff.copy();assert dff.flags.owndata;dff[:]=123;assert m.getDFFValues().tobytes()==original.tobytes()
# Rest fixtures exercise reset; nonrest fixtures preserve the explicitly chosen initial state.
h=simulate_state_history(m,t,sp,reset=f['reset']);assert h.state_M.tobytes()==a.tobytes() and m.getDFFValues().tobytes()==original.tobytes()
assert h.state_M.flags.owndata and h.time_s.flags.owndata and h.state_order==tuple(['G'+str(i) for i in range(9)]+['BCa','Ca','Ca_in'])
assert h.buffer_approximation=='linear' and h.integration_method=='integrateOverTime' and h.internal_dt_s==1e-4
h.state_M[:]=456;assert states().tobytes()==a.tobytes();assert bt==t.tobytes() and bs==sp.tobytes()
assert nr==pickle.dumps(np.random.get_state()) and pr==pickle.dumps(random.getstate());assert files=={str(x):x.stat().st_size for x in pathlib.Path.cwd().rglob('*') if x.is_file()}
print('RESULT '+json.dumps({'id':id,'state':a.tobytes().hex(),'fluorescence':original.tobytes().hex(),'reset':f['reset']}))
