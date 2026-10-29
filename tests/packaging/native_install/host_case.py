import sys,pathlib,json,os,ctypes,importlib
variant,order,profile,case,build,source,config,gpfile,cpfile,trapfile,readyfd=sys.argv[1:]
assert 'numpy' not in sys.modules
trap=None
if profile=='collision':
 trap=ctypes.CDLL(trapfile,mode=os.RTLD_NOW|os.RTLD_GLOBAL)
 trap.g2ax_count.restype=ctypes.c_ulong;trap.g2ax_handler.restype=ctypes.c_void_p
 # Positive control proves actual global lookup reaches our handler, not just that the DSO exists.
 global_=ctypes.CDLL(None);assert ctypes.cast(global_.dgemv_,ctypes.c_void_p).value==ctypes.cast(trap.dgemv_,ctypes.c_void_p).value
 trap.g2ax_arm(1);global_.dgemv_();trap.g2ax_arm(0);assert trap.g2ax_count()==1
 host_handler=trap.g2ax_handler()
sys.path.insert(0,source)
if order=='numpy_first':import numpy as np
import c_spikes.pgas
# Installed public import only; no build-tree injection.
assert pathlib.Path(c_spikes.pgas.__file__).resolve()==pathlib.Path(source).resolve()/'c_spikes/pgas/__init__.py'
m=importlib.import_module('c_spikes.pgas.pgas_bound_cpu')
if order=='extension_first':
 assert 'numpy' not in sys.modules,'extension imports NumPy; requested order unavailable'
 import numpy as np
assert bool(hasattr(m,'CheckedContext'))==(variant=='checked')
gp=np.loadtxt(gpfile,dtype=np.float64);cp=np.loadtxt(cpfile,dtype=np.float64)
assert pathlib.Path(m.__file__).resolve().parent==pathlib.Path(source).resolve()/'c_spikes/pgas'
os.write(int(readyfd),b'READY\n');os.close(int(readyfd));assert sys.stdin.readline()=='GO\n'
A=np.array([[1.,2.],[3.,4.]]);B=np.array([[2.,1.],[0.,2.]])
expected=np.array([[2.,5.],[6.,11.]])
assert np.array_equal(A@B,expected);before_host=(A@B).tobytes();raw=(gp.tobytes(),cp.tobytes())
result={'case':case,'variant':variant}
def make():
 g=m.GCaMP(gp,cp);g.setTimeStepMode(m.TimeStepMode.FIXEDLA);return g
def state(g):return {k:{'shape':v.shape,'bytes':v.tobytes().hex()} for k,v in g.getStates().items()}
if case=='H1':
 g=make();t=np.array([0.,.001,.002]);sp=np.array([.001]);g.integrateOverTime(t,sp);result['states']=state(g);result['dff']=g.getDFFValues().tobytes().hex()
elif case=='H2':
 g=make();x=g.steady_state(1e-6);y=g.steady_state(1e-6);assert x.shape==(12,) and x.flags.owndata and np.isfinite(x).all() and x.tobytes()==y.tobytes() and not np.shares_memory(x,y)
 result['equilibrium']=x.tobytes().hex()
 try:g.steady_state(float('nan'));raise AssertionError('missing nonfinite error')
 except RuntimeError as ex:result['error']=str(ex);assert 'null(): svd failed' in str(ex)
elif case=='H3':
 t=np.array([0.,.001,.002]);data=np.zeros(3);gt=np.array([0.])
 a=m.Analyzer(time=t,data=data,constants_file=config,output_folder='output',column=0,tag='test',niter=2,trainedPriorFile='',append=False,trim=1,verbose=False,gtSpikes=gt,has_trained_priors=False,has_gtspikes=False,maxlen=0,Gparam_file=gpfile,seed=17)
 assert a.run() is None;x=a.get_parameter_samples();assert x.shape==(2,6) and x.flags.owndata and np.isfinite(x).all();result['values']=x.tobytes().hex();result['files']={f.name:f.read_bytes().hex() for f in sorted(pathlib.Path('output').iterdir()) if f.is_file()}
elif case=='H4':
 if variant=='checked':
  ctx=m.CheckedContext.owned_isolated_process();r,g=m.try_create_gcamp(gp,cp,ctx);assert r.committed and r.status==m.CheckedStatus.success and g is not None
  before=m._checked_snapshot(g)
  try:
   m._checked_fault(1,m.CheckedStatus.backend_failed);r=g.try_init(ctx);assert not r.committed and r.status==m.CheckedStatus.backend_failed and before==m._checked_snapshot(g)
  finally:m._checked_clear_fault()
  r,none=m.try_create_gcamp(gp,cp,m.CheckedContext());assert not r.committed and none is None and r.status==m.CheckedStatus.context_required
  result['contract']='checked success/failure/rollback/context'
 else:
  assert not hasattr(m,'try_create_gcamp');g=make();x=g.steady_state(0.);assert x.flags.owndata and x.shape==(12,);result['values']=x.tobytes().hex();result['contract']='default constructor/owned result/checked absence'
else:raise AssertionError(case)
assert raw==(gp.tobytes(),cp.tobytes());assert (A@B).tobytes()==before_host
if trap:assert trap.g2ax_count()==1 and trap.g2ax_handler()==host_handler,'host numerical state changed'
print('RESULT '+json.dumps(result,sort_keys=True))
