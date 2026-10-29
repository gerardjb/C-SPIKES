import struct,math,sys,bisect
EPS=sys.float_info.epsilon;TINY=sys.float_info.min
def raw(a):return bytes.fromhex(a['bytes'])
def vals(a):
 assert a['dtype']=='<f8';b=raw(a);return list(struct.unpack('<'+'d'*(len(b)//8),b))
def same_prefix(a,b,n):assert raw(a)==raw(b)[:n*8], 'prefix bytes differ'
def compare(b,c):
 bo,co=b['observation'],c['observation'];br,cr=b['result'],c['result'];g=vals(bo['grid']);h=vals(co['grid']);t=vals(co['requested']);sp=vals(co['spikes']);s=vals(co['pre_states']);y=vals(cr['state']);n=len(g)
 assert br['time']==cr['time'] and bo['requested']==co['requested'] and bo['spikes']==co['spikes']
 assert h[-1]>=t[-1] and all(h[i]<h[i+1] for i in range(len(h)-1)), 'coverage/representability'
 same_prefix(bo['grid'],co['grid'],n);same_prefix(bo['calcium_input'],co['calcium_input'],n);same_prefix(bo['pre_states'],co['pre_states'],12*n)
 assert co['pre_states']['shape']==[len(h),12] and all(map(math.isfinite,s))
 for i in range(n,len(h)):assert h[i]==t[0]+i*co['fine_dt'], 'new point off lattice'
 # Existing spike predicate must assign on the added nodes as well; amplitude
 # inferred only from baseline's pinned DCaT, supplied by the test runner.
 for i,x in enumerate(h):
  expected=any(abs(x-a)<=co['fine_dt']/2 for a in sp)
  actual=vals(co['calcium_input'])[i]
  assert actual==(compare.spike_amplitude if expected else 0.0), 'spike routing/amplitude'
 if g[-1]>=t[-1]:assert b==c, 'covered case changed'
 else:
  for row,x in enumerate(t):
   if g[0]<=x<=g[-1]:assert raw(br['state'])[row*96:(row+1)*96]==raw(cr['state'])[row*96:(row+1)*96], 'supported returned row changed'
 assert all(map(math.isfinite,y)), 'nonfinite repaired state'
 for row,x in enumerate(t):
  j=bisect.bisect_left(h,x)
  if j<len(h) and h[j]==x:
   assert raw(cr['state'])[row*96:(row+1)*96]==raw(co['pre_states'])[j*96:(j+1)*96], 'grid sample mismatch'
  else:
   assert 0<j<len(h)
   left,right=h[j-1],h[j];weight=(x-left)/(right-left)
   for k in range(12):
    a,z=s[(j-1)*12+k],s[j*12+k];expected=(1-weight)*a+weight*z;actual=y[row*12+k]
    bound=32*EPS*max(abs(a),abs(z),abs(expected),TINY)
    assert abs(actual-expected)<=bound, ('interpolation',row,k,actual,expected,bound)
 return {'baseline_grid_rows':n,'candidate_grid_rows':len(h),'baseline_grid_end':g[-1],'candidate_grid_end':h[-1],'requested_end':t[-1],'changed':b!=c,'prefix_exact':True,'interpolation_checked':True}
