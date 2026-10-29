import pathlib,sys,json,numpy as np
case,build,gpfile,cpfile=sys.argv[1:];sys.path.insert(0,build)
import pgas_bound_cpu as m
gp=np.loadtxt(gpfile,dtype=np.float64);cp=np.loadtxt(cpfile,dtype=np.float64);ctx=m.CheckedContext.owned_isolated_process();raw_before=gp.tobytes(),cp.tobytes()
def make():
 r,g=m.try_create_gcamp(gp,cp,ctx);assert r.committed and r.status==m.CheckedStatus.success and g is not None
 return g
def la(g):
 before=m._checked_snapshot(g);g.setTimeStepMode(m.TimeStepMode.FIXEDLA);assert before==m._checked_snapshot(g)
if case=='C03':
 g=make();la(g);before=m._checked_snapshot(g);g.setTimeStepMode(m.TimeStepMode.FIXED);assert before!=m._checked_snapshot(g)
 for action in [lambda:g.try_set_params(*cp,ctx),lambda:g.try_set_state(1e-6,ctx),lambda:g.try_init(ctx),lambda:g.try_set_params_and_init(*cp,ctx),lambda:g.try_steady_state(0.,ctx)[0]]:
  r=action();assert r.committed and r.status==m.CheckedStatus.success
  before=m._checked_snapshot(g);g.setTimeStepMode(m.TimeStepMode.FIXED);assert before==m._checked_snapshot(g)
elif case=='C04':
 try:
  for n in range(1,5):
   m._checked_fault(n,m.CheckedStatus.backend_failed);r,g=m.try_create_gcamp(gp,cp,ctx);assert not r.committed and r.status==m.CheckedStatus.backend_failed and r.solve_count==n and g is None
 finally:m._checked_clear_fault()
 r,g=m.try_create_gcamp(gp,cp,m.CheckedContext());assert not r.committed and r.status==m.CheckedStatus.context_required and g is None
 la(make())
else:raise AssertionError(case)
assert raw_before==(gp.tobytes(),cp.tobytes())
print('RESULT '+json.dumps({'id':case,'ok':True}))
