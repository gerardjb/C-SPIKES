import sys,importlib.util,json,numpy as np,gc
extension,case,gpfile,cpfile=sys.argv[1:]
spec=importlib.util.spec_from_file_location('pgas_bound_cpu',extension);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
context=mod.CheckedContext.owned_isolated_process();gp=np.loadtxt(gpfile);cp=np.loadtxt(cpfile)
legacy={'setParams','setGmat','flux','steady_state','init','setState','setTimeStepMode','evolve','fixedStep','fixedStep_LA','getDFF','getAmplitude','integrateOverTime','getPoolTotals','getDFFValues','getGValues','getStates'}
assert {x for x in dir(mod.GCaMP) if not x.startswith('_')}==legacy|{'try_set_params','try_set_state','try_init','try_set_params_and_init','try_steady_state'}
assert {x for x in dir(mod.Analyzer) if not x.startswith('_')}=={'run','add_parameter_sample','get_parameter_samples','get_run_status','get_run_message'}
assert {x for x in dir(mod) if not x.startswith('_')}=={'GCaMP','Analyzer','TimeStepMode','CheckedContext','CheckedStatus','CheckedReport','try_create_gcamp'}
if case=='T12':
 gs=np.ascontiguousarray(gp[::-1])[::-1];cs=np.ascontiguousarray(cp[::-1])[::-1]
 assert gs.strides[0]<0 and cs.strides[0]<0 and np.array_equal(gs,gp) and np.array_equal(cs,cp)
 report,m=mod.try_create_gcamp(gs,cs,context);assert report.status==mod.CheckedStatus.success
 before=mod._checked_snapshot(m);gs[:]=0;cs[:]=0;assert mod._checked_snapshot(m)==before
elif case=='T13':
 report,m=mod.try_create_gcamp(gp,cp,context);assert report.status==mod.CheckedStatus.success
 mod._checked_seed_histories(m);report,state=m.try_steady_state(1e-6,context);assert report.status==mod.CheckedStatus.success
 histories=m.getStates();saved={k:v.copy() for k,v in histories.items()};saved_state=state.copy()
 assert all(v.flags.owndata for v in histories.values()) and state.flags.owndata
 assert m.try_set_params_and_init(3e-5,1000.,1e-5,5.,30.,10.,context).status==mod.CheckedStatus.success
 assert all(np.array_equal(v,saved[k]) for k,v in histories.items()) and np.array_equal(state,saved_state)
 histories['G'][0,0]=999;assert m.getStates()['G'][0,0]!=999
 histories['G'][0,0]=saved['G'][0,0];del m;gc.collect()
 assert all(np.array_equal(v,saved[k]) for k,v in histories.items()) and np.array_equal(state,saved_state)
elif case=='T14':
 report,m=mod.try_create_gcamp(gp,cp,context);assert report.status==mod.CheckedStatus.success
 alias=m;before=mod._checked_snapshot(m);report=m.try_set_state(1e-6,context)
 assert alias is m and report.status==mod.CheckedStatus.success and report.committed
 assert before!=mod._checked_snapshot(alias);before=mod._checked_snapshot(m)
 mod._checked_fault(1,mod.CheckedStatus.nonunique);report,state=alias.try_steady_state(0.,context)
 assert report.status==mod.CheckedStatus.nonunique and not report.committed and state is None
 assert before==mod._checked_snapshot(m)==mod._checked_snapshot(alias);mod._checked_clear_fault()
else:raise AssertionError('unfrozen Python case')
print('RESULT '+json.dumps({'id':case,'ok':True}))
