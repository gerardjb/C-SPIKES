from pathlib import Path
import json,os,sys,hashlib
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir();sys.path.insert(0,str(R));from processes import execute
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();refs=json.loads((R/'fixtures/host_reference.json').read_text());ledger=[];prior_place='A'
def save():(O/'LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
try:
 for spec in json.loads((R/'MATRIX.json').read_text()):
  place=spec['place'];v=spec['variant'];name=spec['id'];kind=spec['kind'];case=spec['case']
  if place!=prior_place:
   assert place=='B'and not(R/'B').exists();(R/'A').rename(R/'B');assert not(R/'A').exists();prior_place=place
  root=R/place/v;cwd=O/(name+'-cwd');cwd.mkdir();row={'id':name,'status':'started'};ledger.append(row);save()
  if kind=='host':args=[sys.executable,R/'host_case.py',v,spec['order'],spec['profile'],case,root/'c_spikes/pgas',root,R/'pipeline_0.json',R/'gparams.dat',R/'cparams.dat',R/'collision.so']
  elif kind=='oasis':args=[sys.executable,R/'oasis_case.py',root,case]
  else:args=[sys.executable,R/'package_case.py',root,v,case]
  stdout,timing=execute([str(x)for x in args],cwd,O,name,ready=True,body_limit=10);row.update(timing)
  lines=[x[7:]for x in stdout.splitlines()if x.startswith('RESULT ')];assert len(lines)==1;result=json.loads(lines[0])
  if kind=='host':assert result==refs[spec['oracle']],name
  else:assert result['passed']and result['id']==case
  libraries=json.loads((O/(name+'.libraries.json')).read_text());assert not any(str(R/'source')in p or str(R/'build')in p or '/vcpkg/'in p for p in libraries)
  own=[p for p in libraries if 'pgas_bound_cpu' in p or 'oasis_methods' in p];assert own and all(Path(p).is_relative_to(root)for p in own)
  row['status']='passed';save()
 assert len(ledger)==80 and all(x['status']=='passed'for x in ledger)
 for v in ['default','checked']:
  root=R/'B'/v;manifest=json.loads((root/'share/c_spikes/ARTIFACT_MANIFEST.json').read_text())
  for n,h in manifest['files'].items():assert sha(root/n)==h,n
 (O/'SUMMARY.json').write_text(json.dumps({'passed':80,'installed':40,'relocated':40,'old_prefix_absent':not(R/'A').exists()},indent=2)+'\n')
except BaseException as ex:
 if ledger and ledger[-1]['status']=='started':ledger[-1]['status']='failed_stop';ledger[-1]['error']=str(ex)
 save();(O/'STOP.json').write_text(json.dumps({'error':str(ex),'passed':sum(x['status']=='passed'for x in ledger)},indent=2)+'\n');raise
