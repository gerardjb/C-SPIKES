import sys,os,pathlib,json,hashlib,pickle,random,logging,copy
import numpy as np
R=pathlib.Path(sys.argv[1]);group,case,variant=sys.argv[2:5];ready=int(sys.argv[5])
plan=json.loads((R/'cases.json').read_text());pkg=R/('runtime_'+variant)
sys.path.insert(0,str(pkg/'src'))
from c_spikes.pgas import pgas_bound as pgas
from c_spikes.pgas import pgas_bound_cpu
from c_spikes.syn_gen import simulate_state_history
import c_spikes.syn_gen.state_history as sh
from c_spikes.syn_gen.synth_gen import synth_gen
import scipy.io as sio
if case=='W07':
 from c_spikes.cascade2p.utils import load_recordings_from_file as cascade_read
if case=='W08':
 from c_spikes.ens2.ENS2 import load_recordings_from_file as ens_read
assert pgas.__backend__=='cpu'
expected=json.loads((R/'RUNTIME_PINS.json').read_text())[variant]
assert hashlib.sha256(pathlib.Path(pgas_bound_cpu.__file__).read_bytes()).hexdigest()==expected
G=np.loadtxt(R/'gparams.dat');C=np.loadtxt(R/'cparams.dat').reshape(-1)
config=json.loads((R/'config.json').read_text())
os.write(ready,b'READY\n');os.close(ready);assert sys.stdin.readline()=='GO\n'
result={'id':case,'variant':variant,'ok':True};cwd=pathlib.Path.cwd()
def array(a):
 a=np.asarray(a);return {'shape':list(a.shape),'dtype':str(a.dtype),'hex':a.tobytes().hex()}
def model(c=None):
 m=pgas.GCaMP(G,C if c is None else c);m.setTimeStepMode(pgas.TimeStepMode.FIXEDLA);return m
def snapshot(root):return {str(p.relative_to(root)):p.read_bytes().hex() for p in root.rglob('*') if p.is_file()}
HEADER='G_tot,gamma,DCaT,Rf,gam_in,gam_out,sigma2,r0,r1,w01,w10\n'
VALUES=[6.04700454e-5,1000.,1e-5,5.,40.,4.,.01,.2,.3,.4,.5]
def saved(root,rows=1):
 root.mkdir();(root/'param_samples_t.dat').write_text(HEADER+(','.join(map(str,VALUES))+'\n')*rows)
 (root/'traj_samples_t.dat').write_text('index,burst,B,S,C,Y\n'+''.join(f'{i},0,0,0,0,0\n' for i in range(rows) for _ in range(3)))
 (root/'logp_t.dat').write_text('0\n'*rows);(root/'sentinel').write_text('preserve')
def analyze(root,niter,trim=1,append=False,seed=17):
 cfg=cwd/'config.json';cfg.write_text(json.dumps(config));a=pgas.Analyzer(np.array([0.,.001,.002]),np.zeros(3),str(cfg),str(root),1,'t',niter,'',append,trim,False,np.zeros(0),False,False,0,str(R/'gparams.dat'),seed);a.run();return a
def check_rows(root,n):
 rows=(root/'param_samples_t.dat').read_text().splitlines();assert rows[0]+'\n'==HEADER and len(rows)==n+1
 traj=(root/'traj_samples_t.dat').read_text().splitlines();assert len(traj)==1+3*n
 assert [int(x.split(',')[0]) for x in traj[1:]]==[i for i in range(n) for _ in range(3)]
 assert len((root/'logp_t.dat').read_text().splitlines())==n
 return rows
if group=='append':
 root=cwd/'out'
 if case in ('A01','A02'):
  n=2 if case=='A01' else 3;a=analyze(root,n,seed=17 if case=='A01' else 29);check_rows(root,n)
  result['compare']={'files':snapshot(root),'parameter_samples':array(a.get_parameter_samples())}
  if variant!='baseline':assert a.get_run_status()=='completed'
 elif case in ('A03','A05'):
  rows=1 if case=='A03' else 2;saved(root,rows);before=snapshot(root);a=analyze(root,rows+1,trim=1 if case=='A03' else 2,append=True);out=check_rows(root,rows+1)
  assert a.get_run_status()=='completed';assert np.array_equal(np.array(out[-1].split(','),float),[float(format(x,'.6g')) for x in VALUES])
  for n,h in before.items():assert (root/n).read_bytes().startswith(bytes.fromhex(h))
  assert np.array_equal(a.get_parameter_samples(),np.array([VALUES[:6]]))
  result['restored']=VALUES;result['rows']=rows+1
 elif case=='A04':
  a=analyze(root,3,trim=2);check_rows(root,3);assert a.get_run_status()=='completed';result['rows']=3
 else:
  if case!='A07':saved(root)
  if case=='A08':(root/'param_samples_t.dat').write_text(HEADER)
  if case=='A09':(root/'param_samples_t.dat').write_text(HEADER+'1,2\n')
  if case=='A10':(root/'param_samples_t.dat').write_text('G_tot,gamma,DCaT,Rf,sigma2,r0,r1,w01,w10\n'+','.join(map(str,VALUES[:9]))+'\n')
  if case=='A11':(root/'param_samples_t.dat').write_text(HEADER+'bad,'+','.join(map(str,VALUES[1:]))+'\n')
  if case=='A12':(root/'traj_samples_t.dat').unlink()
  if case=='A13':(root/'traj_samples_t.dat').write_text('index,burst,B,S,C,Y\n9,0,0,0,0,0\n')
  before=snapshot(root);a=analyze(root,1 if case=='A06' else 2,trim=0 if case=='A14' else 1,append=True)
  assert a.get_run_status()==('no_op' if case=='A06' else 'invalid_restart')
  assert a.get_run_message() and snapshot(root)==before
  if case=='A07':assert not root.exists()
  result['status']=a.get_run_status();result['files_unchanged']=True
elif group=='occupancy':
 idx=int(case[1:]);f=plan['fixtures'][(idx-1)%4] if idx<=4 else plan['fixtures'][0]
 t=np.array([float.fromhex(x) for x in f['time_hex']]);sp=np.array([float.fromhex(x) for x in f['spikes_hex']]);ca=float.fromhex(f['initial_ca_hex'])
 m=model();m.setState(ca)
 if case=='O05':m.setParams(3e-5,*C[1:]);m.setState(ca)
 nr=pickle.dumps(np.random.get_state());pr=pickle.dumps(random.getstate());before=snapshot(cwd)
 if variant=='baseline':h=simulate_state_history(m,t,sp,reset=f['reset'])
 else:
  from c_spikes.syn_gen import simulate_state_occupancies
  o=simulate_state_occupancies(m,t,sp,reset=f['reset']);h=o.history
  assert o.valid_denominators==(True,True) and o.pool_totals_M==(3e-5 if case=='O05' else float(C[0]),.004)
  assert np.array_equal(o.indicator_fraction,h.state_M[:,:9]/o.pool_totals_M[0],equal_nan=True)
  assert np.array_equal(o.buffer_bound_fraction,h.state_M[:,9]/.004,equal_nan=True)
  assert not np.shares_memory(o.indicator_fraction,h.state_M) and not np.shares_memory(o.buffer_bound_fraction,h.state_M)
  assert np.isfinite(h.state_M).all();assert np.allclose(o.indicator_fraction.sum(axis=1),1,rtol=1e-10,atol=1e-13)
  if case=='O03':
   old=h.state_M.copy()
   try:simulate_state_occupancies(m,np.array([1.,1.]),sp)
   except ValueError:pass
   else:raise AssertionError('invalid times accepted')
   assert np.array_equal(old,h.state_M)
  if case in ('O04','O06'):
   old=h.state_M.copy();frac=o.indicator_fraction.copy();tot=o.pool_totals_M
   m.setParams(float(C[0])*2,*C[1:]);other=simulate_state_occupancies(m,t,sp,reset=True)
   assert other.pool_totals_M[0]==float(C[0])*2 and o.pool_totals_M==tot
   other.history.state_M[:]=123;other.indicator_fraction[:]=456
   assert np.array_equal(h.state_M,old) and np.array_equal(o.indicator_fraction,frac)
  if case=='O07':
   for bad in (0.,-1.,float('nan'),float('inf')):
    q=sh._occupancies_from_history(h,(bad,.004));assert q.valid_denominators==(False,True) and np.isnan(q.indicator_fraction).all()
    q=sh._occupancies_from_history(h,(float(C[0]),bad));assert q.valid_denominators==(True,False) and np.isnan(q.buffer_bound_fraction).all()
  if case=='O08':
   raw=h.state_M.copy();raw[0,0]=np.nan;raw[1,9]=np.inf;raw[2,1]=-1
   hh=sh.StateHistory(h.time_s.copy(),raw);q=sh._occupancies_from_history(hh,(float(C[0]),.004))
   assert np.isnan(q.indicator_fraction[0,0]) and np.isposinf(q.buffer_bound_fraction[1]) and q.indicator_fraction[2,1]<0
   assert np.isnan(hh.state_M[0,0]) and np.isposinf(hh.state_M[1,9])
 result['compare']={'time':array(h.time_s),'state':array(h.state_M)}
 assert nr==pickle.dumps(np.random.get_state()) and pr==pickle.dumps(random.getstate()) and before==snapshot(cwd)
else:
 # Fixtures contain real MATLAB structs and are written/read by the actual public writer.
 noise_dir=cwd/'noise';noise_dir.mkdir()
 def mat(path,t,noise,fields=True):
  names=['gt_noise','marker']+(['fluo_mean','events_AP','fluo_time'] if fields else [])
  inner=np.zeros((1,1),dtype=[(n,'O') for n in names]);inner['gt_noise'][0,0]=np.array(noise);inner['marker'][0,0]=np.array([[19,23]])
  if fields:inner['fluo_mean'][0,0]=np.zeros(len(t));inner['events_AP'][0,0]=np.zeros((0,1));inner['fluo_time'][0,0]=np.array(t)
  outer=np.empty((1,1),dtype=object);outer[0,0]=inner;sio.savemat(path,{'CAttached':outer})
 def contents(path):
  outer=sio.loadmat(path)['CAttached'];res=[]
  for cell in outer[0]:
   inner=cell[0,0];res.append({n:array(inner[n]) for n in inner.dtype.names})
  return res
 class Injected:
  def __init__(self,nonfinite=False):self.calls=[];self.resets=0;self.nonfinite=nonfinite
  def setParams(self,*v):self.calls.append(tuple(v))
  def init(self):self.resets+=1
  def integrateOverTime(self,t,sp):self.t=np.array(t);self.sp=np.array(sp)
  def getDFFValues(self):
   y=np.linspace(.1,.2,len(self.t));
   if self.nonfinite:y[0]=np.nan;y[1]=np.inf
   return y
 times=np.array([1.,1.0001,1.0002,1.0003,1.0004]);noise=np.arange(5)*.01
 if case=='W11':times[2]=np.nan
 if case=='W03':noise[2]=np.nan
 mat(noise_dir/'a.mat',times,noise,fields=case!='W12')
 if case in ('W01','W02','W04'):mat(noise_dir/'b.mat',times,noise+.03)
 injected=case in ('W03','W04','W10','W11','W12');m=Injected(case=='W03') if injected else model()
 kwargs=dict(cell_params=C.tolist(),GCaMP_model=m,noise_dir=str(noise_dir),tag='t',noise_seed=17 if case!='W02' else 29,seed_spikes=True,use_noise=True,log_nonfinite=True)
 if case=='W04':kwargs.update(noise_fraction=.5,noise_seed=[17,29])
 if case=='W05':kwargs.pop('GCaMP_model')
 s=synth_gen(**kwargs)
 if case=='W11':s.spk_gen=lambda T:np.array([1.0001,np.nan])
 if case in ('W07','W08'):
  # Reader-boundary fixture: prescribed spikes/state output, no new scientific simulation.
  times=np.arange(181,dtype=float)/60;noise=np.zeros(181);mat(noise_dir/'a.mat',times,noise)
  s.gcamp=Injected();s.spk_gen=lambda T:np.array([.8,1.2,1.8,2.2])
 if case=='W09':
  from c_spikes.syn_gen import build_synthetic_ground_truth_from_pgas
  # The wrapper reads parameter samples before its output guard; supply its current six-column cache shape.
  cache=cwd/'params.dat';cache.write_text(HEADER+(','.join(map(str,VALUES))+'\n')*2)
  target=cwd/'existing'/'Ground_truth'/'synth_t';target.mkdir(parents=True);(target/'keep.mat').write_bytes(b'untouched');before=snapshot(target)
  try:build_synthetic_ground_truth_from_pgas(cache,gparam_path=R/'gparams.dat',burnin=0,noise_dir=noise_dir,tag='t',output_root=cwd/'existing')
  except FileExistsError:pass
  else:raise AssertionError('overwrite guard not reached')
  assert snapshot(target)==before
 else:
  override=C.tolist() if case=='W02' else (C.copy() if case in ('W05','W06') else None)
  s.generate(Cparams=override,output_folder=str(cwd/'generated'))
  paths=sorted((cwd/'generated'/'Ground_truth'/'synth_t').glob('*.mat'));assert paths
  parsed=[contents(p) for p in paths];result['compare']={'contents':parsed,'names':[p.name for p in paths],'rng':pickle.dumps(np.random.get_state()).hex()}
  for p in paths:
   inner=sio.loadmat(p)['CAttached'][0][0][0,0]
   assert {'fluo_mean','fluo_time','events_AP','gt_noise','marker'}<=set(inner.dtype.names)
   assert np.array_equal(inner['marker'],[[19,23]])
   if case=='W07':
    got=cascade_read(str(p));assert len(got)==1 and np.array_equal(got[0]['t'],times)
    assert np.array_equal(got[0]['spikes'].ravel(),np.array([.8,1.2,1.8,2.2]))
    assert np.array_equal(got[0]['dff'],inner['fluo_mean'].ravel())
   if case=='W08':
    neuron,trials,n=ens_read(str(p),60);assert len(trials)==1 and n>0 and np.isfinite(trials[0]['trace_seq']).all()
   if case=='W11':assert np.isfinite(inner['fluo_time']).all() and np.isfinite(inner['events_AP']).all()
  if injected:assert m.resets>=2 and all(np.array_equal(v,C) for v in m.calls)
  if case=='W10':
   assert np.array_equal(s.Cparams,C)
   try:s.generate(Cparams=np.array([]),output_folder=str(cwd/'empty'))
   except (IndexError,ValueError,TypeError):pass
   else:raise AssertionError('explicit empty override silently fell back')
  if case=='W05':
   s2=synth_gen(cell_params=C.copy(),noise_dir=str(noise_dir),tag='t',noise_seed=17,seed_spikes=True,use_noise=True)
   s2.generate(Cparams=C.tolist(),output_folder=str(cwd/'again'))
   assert [contents(p) for p in sorted((cwd/'again'/'Ground_truth'/'synth_t').glob('*.mat'))]==parsed
  if case=='W06':
   s.generate(Cparams=C.tolist(),output_folder=str(cwd/'again'));assert [contents(p) for p in sorted((cwd/'again'/'Ground_truth'/'synth_t').glob('*.mat'))]==parsed
print('RESULT '+json.dumps(result,allow_nan=False))
