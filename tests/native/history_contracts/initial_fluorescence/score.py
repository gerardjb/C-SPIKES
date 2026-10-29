import struct,math,sys,bisect
EPS=sys.float_info.epsilon;TINY=sys.float_info.min
def vals(h):return struct.unpack('<'+'d'*(len(h)//16),bytes.fromhex(h))
def close(a,b,scale):
 assert math.isfinite(a) and math.isfinite(b)
 assert abs(a-b)<=32*EPS*max(abs(a),abs(b),scale,TINY),('rounding bound',a,b)
def candidate(c):
 t=vals(c['time']);g=vals(c['grid']);y=vals(c['fluorescence']);f=(vals(c['initial_formula'])[0],)+vals(c['pre_fluorescence_tail']);assert len(g)==len(f)
 scale=vals(c['formula_scale'])[0];close(y[0],f[0],scale)
 for x,a in zip(t,y):
  j=bisect.bisect_left(g,x)
  if j<len(g) and g[j]==x:close(a,f[j],scale if j==0 else abs(f[j]))
  else:
   assert 0<j<len(g);weight=(x-g[j-1])/(g[j]-g[j-1]);b=(1-weight)*f[j-1]+weight*f[j];close(a,b,max(abs(f[j-1]),abs(f[j]),scale if j==1 else 0))
 return {'initial':y[0],'initial_formula':f[0],'all_finite':all(map(math.isfinite,y)),'interpolation_checked':True}
def compare(b,c):
 for k in ['time','state','stored_state','grid','input','pre_states','pre_fluorescence_tail','later_fluorescence','initial_formula','formula_scale']:assert b[k]==c[k],('baseline/candidate mismatch',k)
 return candidate(c)
