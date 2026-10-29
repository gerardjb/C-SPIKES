from pathlib import Path
import json,subprocess,hashlib,re
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof={}
for v in ['default','checked']:
 b=R/'build'/v;f=next(b.glob('pgas_bound_cpu*.so'));installed=next((R/'A'/v/'c_spikes/pgas').glob('pgas_bound_cpu*.so'))
 sy=subprocess.check_output(['readelf','-Ws',str(installed)],text=True);dy=subprocess.check_output(['readelf','-d',str(installed)],text=True);ds=subprocess.check_output(['readelf','--dyn-syms','-W',str(installed)],text=True);rel=subprocess.check_output(['readelf','-Wr',str(installed)],text=True);mp=(b/'pgas_bound_cpu.map').read_text();link=(b/'CMakeFiles/pgas_bound_cpu.dir/link.txt').read_text()
 assert 'libarmadillo'not in dy+mp and 'wrapper2_'not in sy
 assert 'libopenblas.a'in mp and 'liblapack.a'in mp and 'libgfortran.so.3'in dy
 assert 'csp-provider/openblas-xerbla.c.o'in mp and 'liblapack.a(xerbla'not in mp
 assert '--version-script'in link and '--undefined'not in link and '--whole-archive'not in link and '-Bsymbolic'not in link
 rows=[l.split()for l in ds.splitlines()if len(l.split())>=8 and l.split()[0].rstrip(':').isdigit()]
 assert {l[7].split('@')[0]for l in rows if l[6]!='UND'and l[4]in['GLOBAL','WEAK']}=={'PyInit_pgas_bound_cpu'}
 numerical=lambda n:n.startswith(('gsl_','cblas_','openblas_'))or n in ['dgemv_','dgesdd_','xerbla_','__xerbla']
 assert not any(numerical(l[7].split('@')[0])for l in rows)
 assert not re.search(r'\b(?:gsl_\w+|cblas_\w+|dgemv_|dgesdd_|xerbla_|__xerbla)\b',rel)
 for symbol in ['dgemv_','dgesdd_','xerbla_','gsl_rng_uniform','gsl_set_error_handler']:
  lines=[l.split()for l in sy.splitlines()if l.strip().endswith(' '+symbol)];assert lines and all(l[4]=='LOCAL'and l[6]!='UND'for l in lines),symbol
 xer=[l.split()for l in sy.splitlines()if l.strip().endswith(' xerbla_')];alias=[l.split()for l in sy.splitlines()if l.strip().endswith(' __xerbla')];assert xer and alias and all(l[2]=='30'for l in xer+alias)and alias[0][1]==xer[0][1]
 if v=='checked':assert 'libgslcblas.a'in mp and 'cblas_dgemv'in sy and 'libopenblas.a(cblas_'not in mp
 for module in [installed,next((R/'A'/v/'c_spikes/oasis').glob('oasis_methods*.so'))]:
  dynamic=subprocess.check_output(['readelf','-d',str(module)],text=True);assert not any(x in dynamic for x in ['RPATH','RUNPATH','libarmadillo'])
  ldd=subprocess.check_output(['ldd',str(module)],text=True);assert 'not found'not in ldd
  (R/'proofs'/(v+'-'+module.name+'.dynamic')).write_text(dynamic);(R/'proofs'/(v+'-'+module.name+'.ldd')).write_text(ldd)
  for line in ldd.splitlines():
   parts=line.split();paths=[Path(x)for x in parts if x.startswith('/')]
   for p in paths:
    assert '/vcpkg/'not in str(p)and str(R)not in str(p),str(p)
    proof[str(p.resolve())]={'sha256':sha(p),'kind':'external runtime'}
 for dep in b.glob('CMakeFiles/pgas_obj_cpu.dir/src/c_spikes/pgas/*.o.d'):
  t=dep.read_text();assert 'armadillo_bits'not in t and '/include/armadillo 'not in t
 entries=json.loads((b/'compile_commands.json').read_text());expected={str(x.resolve())for x in (R/'source/src/c_spikes/pgas').glob('*.cpp')};assert len(expected)==14 and expected<={str(Path(x['file']).resolve())for x in entries}
 for n in ['g4a_header','g4a_signatures']:
  deps=list((b/('CMakeFiles/'+n+'.dir')).rglob('*.o.d'));assert len(deps)==1 and 'armadillo_bits'not in deps[0].read_text()
 for suffix,value in [('symbols',sy),('dynamic',dy),('dynsymbols',ds),('relocations',rel),('map',mp),('link',link)]: (R/'proofs'/(v+'.'+suffix)).write_text(value)
 proof[v]={'installed_module':sha(installed),'no_armadillo':True,'module_private':True,'handler_and_cblas':True,'compiled_translation_units':14}
(R/'proofs/PROVIDER_RUNTIME.json').write_text(json.dumps(proof,indent=2)+'\n')
