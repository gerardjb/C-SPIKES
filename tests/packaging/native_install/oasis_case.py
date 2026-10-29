from pathlib import Path
import sys,os,json
prefix,case,fd=sys.argv[1:];root=Path(__file__).resolve().parent;sys.path.insert(0,str(Path(prefix).resolve()));sys.path.insert(0,str(root/'fixtures'))
import test_oasis_parity as test
import c_spikes.oasis.functions as f
assert Path(f.__file__).resolve().is_relative_to(Path(prefix).resolve())
os.write(int(fd),b'READY\n');os.close(int(fd));assert sys.stdin.readline()=='GO\n'
getattr(test,'test_deconvolve_'+case+'_full_output_matches_minimal_port')()
print('RESULT '+json.dumps({'id':case,'passed':True}))
