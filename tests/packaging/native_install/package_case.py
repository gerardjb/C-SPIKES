from pathlib import Path
import sys,os,json,hashlib,importlib
prefix,variant,case,fd=sys.argv[1:];root=Path(prefix).resolve();sys.path.insert(0,str(root))
import c_spikes
import c_spikes.pgas
from c_spikes.pgas import pgas_bound_cpu as m
import c_spikes.oasis
from c_spikes.oasis import functions,oasis_methods
mods=[c_spikes,c_spikes.pgas,m,c_spikes.oasis,functions,oasis_methods]
assert all(Path(x.__file__).resolve().is_relative_to(root)for x in mods)
os.write(int(fd),b'READY\n');os.close(int(fd));assert sys.stdin.readline()=='GO\n'
manifest=json.loads((root/'share/c_spikes/ARTIFACT_MANIFEST.json').read_text())
for n,h in manifest['files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
assert manifest['variant']==variant
assert type(manifest['observations']) is bool
assert m.numerical_diagnostics_enabled is manifest['observations']
if case=='P1':
 assert not any(p.name.startswith('libarmadillo')for p in root.rglob('*'))
 assert not hasattr(c_spikes,'syn_gen')
 assert (root/'share/c_spikes/LICENSE').is_file() and (root/'share/c_spikes/OASIS_PROVENANCE.md').is_file()
 assert len(list((root/'share/c_spikes/notices').glob('*')))==7
elif case=='P2':
 public=lambda x:{n for n in dir(x)if not n.startswith('_')}
 base={'setParams','setGmat','flux','steady_state','init','setState','setTimeStepMode','evolve','fixedStep','fixedStep_LA','getDFF','getAmplitude','integrateOverTime','getDFFValues','getGValues','getStates','getPoolTotals'}
 checked={'try_set_params','try_set_state','try_init','try_set_params_and_init','try_steady_state'}
 assert public(m.GCaMP)==base|(checked if variant=='checked'else set())
 assert public(m.Analyzer)=={'run','add_parameter_sample','get_parameter_samples','get_run_status','get_run_message'}
 assert public(m)=={'GCaMP','Analyzer','TimeStepMode','numerical_diagnostics_enabled'}|({'CheckedContext','CheckedStatus','CheckedReport','try_create_gcamp'}if variant=='checked'else set())
 assert set(m.TimeStepMode.__members__)=={'FIXED','FIXEDLA'}
 assert int(m.TimeStepMode.FIXED)==0 and int(m.TimeStepMode.FIXEDLA)==1
 assert 'time:' in m.GCaMP.integrateOverTime.__doc__ and 'spike_times:' in m.GCaMP.integrateOverTime.__doc__
 assert not hasattr(m.GCaMP,'integrateOverTime2')
 if variant=='checked':
  assert public(m.CheckedReport)=={'status','committed','solve_count','backend_code','diagnostics'}
  assert public(m.CheckedContext)=={'owned_isolated_process'}
  assert set(m.CheckedStatus.__members__)=={'success','input_invalid','backend_failed','malformed_factors','svd_quality_failed','no_nullspace','nonunique','rank_ambiguous','normalization_unsafe','analytic_no_selection','invariant_failed','context_required','context_busy'}
  assert all(hasattr(m,x)for x in ['_checked_snapshot','_checked_fault','_checked_clear_fault'])
 else:assert not hasattr(m,'_checked_snapshot')
else:raise AssertionError(case)
print('RESULT '+json.dumps({'id':case,'variant':variant,'passed':True,'module_files':[str(Path(x.__file__).relative_to(root))for x in mods]}))
