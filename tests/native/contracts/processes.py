import pathlib,json,os,time,subprocess,signal,selectors
DEADLINE=float(os.environ['G2AD_DEADLINE'])
def execute(args,cwd,output,name,ready=False,body_limit=3):
 assert DEADLINE-time.monotonic()>=(32+body_limit if ready else 2+body_limit),'global process admission'
 p=None;rr=rw=None;row={};start=time.monotonic()
 try:
  if ready:rr,rw=os.pipe();args=args+[str(rw)]
  with (output/(name+'.stdout')).open('wb') as so,(output/(name+'.stderr')).open('wb') as se:
   p=subprocess.Popen(args,cwd=cwd,stdout=so,stderr=se,stdin=subprocess.PIPE if ready else subprocess.DEVNULL,start_new_session=True,pass_fds=(rw,) if ready else ())
   if ready:
    os.close(rw);rw=None
    with selectors.DefaultSelector() as sel:
     sel.register(rr,selectors.EVENT_READ);assert sel.select(timeout=30),'startup timeout';assert os.read(rr,32)==b'READY\n','startup exited before READY'
    row['startup_s']=time.monotonic()-start
    maps=pathlib.Path('/proc')/str(p.pid)/'maps'
    libraries=sorted({line.split()[-1] for line in maps.read_text().splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/')})
    (output/(name+'.libraries.json')).write_text(json.dumps(libraries,indent=2)+'\n')
    assert not any('libarmadillo' in path for path in libraries),'candidate loaded Armadillo'
    p.stdin.write(b'GO\n');p.stdin.flush()
   body=time.monotonic();row['returncode']=p.wait(timeout=body_limit);row['body_s']=time.monotonic()-body
  assert row['returncode']==0,name
  return (output/(name+'.stdout')).read_text(),row
 finally:
  if p and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=2)
  if rr is not None:os.close(rr)
  if rw is not None:os.close(rw)
