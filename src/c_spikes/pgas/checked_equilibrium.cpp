#ifdef C_SPIKES_CHECKED_CPU
#include "include/GCaMP_model.h"
#include "include/checked_equilibrium.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <climits>
namespace checked { namespace detail {
namespace {
constexpr double eps=std::numeric_limits<double>::epsilon();
template<class A> bool finite(const A& a){for(double x:a)if(!std::isfinite(x))return false;return true;}
double at(const Matrix9& a,int i,int j){return a[9*i+j];}
double infnorm(const Matrix9& a){double v=0;for(int i=0;i<9;++i){double s=0;for(int j=0;j<9;++j)s+=std::abs(at(a,i,j));v=std::max(v,s);}return v;}
bool valid_input(const Matrix9& q,double mass){return finite(q)&&std::isfinite(mass)&&mass>0;}
}
Factors gsl_backend(const Matrix9& q){
 Factors f;f.u=q;Vector9 workspace{};
 auto u=gsl_matrix_view_array(f.u.data(),9,9);auto v=gsl_matrix_view_array(f.v.data(),9,9);
 auto s=gsl_vector_view_array(f.s.data(),9);auto work=gsl_vector_view_array(workspace.data(),9);
 f.code=gsl_linalg_SV_decomp(&u.matrix,&v.matrix,&s.vector,&work.vector);return f;
}
std::array<int,3> classify(const Vector9& s){
 std::array<int,3> ranks{};const double tau=9*eps*s[0],factors[3]={.1,1,10};
 for(double x:s)for(int k=0;k<3;++k)ranks[k]+=x>tau*factors[k];return ranks;
}
bool normalize(const Vector9& z,double mass,Vector9& output,double& sum,double& guard){
 sum=0;double l1=0;for(double x:z){sum+=x;l1+=std::abs(x);}guard=64*eps*l1;
 if(!finite(z)||!std::isfinite(mass)||mass<=0||!std::isfinite(sum)||!std::isfinite(guard)||!(std::abs(sum)>guard))return false;
 Vector9 g{};for(int i=0;i<9;++i)g[i]=z[i]/sum*mass;
 if(!finite(g))return false;output=g;return true;
}
bool positive(const Vector9& g,double mass){
 if(!std::isfinite(mass)||mass<=0||!finite(g))return false;
 for(double x:g)if(x/mass < -1e-12)return false;return true;
}
Result assess(const Matrix9& q,const Factors& f,bool generator,double mass){
 Result r;if(!valid_input(q,mass))return r;
 r.backend_called=true;r.factors=f;
 if(f.code!=0){r.outcome=Outcome::backend_failed;return r;}
 r.sorted=finite(f.s)&&finite(f.u)&&finite(f.v);
 for(int i=0;i<9;++i)r.sorted=r.sorted&&f.s[i]>=0&&(i==0||f.s[i]<=f.s[i-1]);
 if(!r.sorted){r.outcome=Outcome::malformed_factors;return r;}
 r.tau=9*eps*f.s[0];r.ranks=classify(f.s);r.nullity=9-r.ranks[1];
 Matrix9 reconstruction{},ou{},ov{};
 for(int i=0;i<9;++i)for(int j=0;j<9;++j){
  double x=0,uu=0,vv=0,pp=0;
  for(int k=0;k<9;++k){x+=at(f.u,i,k)*f.s[k]*at(f.v,j,k);uu+=at(f.u,k,i)*at(f.u,k,j);vv+=at(f.v,k,i)*at(f.v,k,j);if(f.s[k]<=r.tau)pp+=at(f.v,i,k)*at(f.v,j,k);}
  reconstruction[9*i+j]=x-at(q,i,j);ou[9*i+j]=uu-(i==j);ov[9*i+j]=vv-(i==j);r.projector[9*i+j]=pp;
 }
 const double qnorm=infnorm(q),error=infnorm(reconstruction);
 r.reconstruction=qnorm>0?error/qnorm:error;r.u_orthogonality=infnorm(ou);r.v_orthogonality=infnorm(ov);r.has_diagnostics=true;
 if(!finite(reconstruction)||!finite(ou)||!finite(ov)||!finite(r.projector)||!std::isfinite(qnorm)||!std::isfinite(r.reconstruction)||!std::isfinite(r.v_orthogonality)||r.reconstruction>1e-12||r.v_orthogonality>1e-12){r.outcome=Outcome::svd_quality_failed;return r;}
 if(r.nullity==0){r.outcome=Outcome::no_nullspace;return r;}
 if(r.nullity>1){r.outcome=Outcome::nonunique;return r;}
 if(r.ranks!=std::array<int,3>{{8,8,8}}){r.outcome=Outcome::rank_ambiguous;return r;}
 int k=0;while(k<9&&f.s[k]>r.tau)++k;
 if(k==9){r.outcome=Outcome::malformed_factors;return r;}
 Vector9 z{},candidate{};for(int i=0;i<9;++i)z[i]=at(f.v,i,k);
 if(!normalize(z,mass,candidate,r.sum,r.guard)){r.outcome=Outcome::normalization_unsafe;return r;}
 if(!generator){r.outcome=Outcome::analytic_no_selection;return r;}
 double total=0,minval=std::numeric_limits<double>::infinity(),maxabs=0,resid=0;
 for(double x:candidate){total+=x;minval=std::min(minval,x);maxabs=std::max(maxabs,std::abs(x));}
 for(int i=0;i<9;++i){double x=0;for(int j=0;j<9;++j)x+=at(q,i,j)*candidate[j];resid=std::max(resid,std::abs(x));}
 const double den=qnorm*maxabs;r.residual=den>0?resid/den:std::numeric_limits<double>::quiet_NaN();
 r.mass_error=std::abs(total-mass)/mass;r.minimum=minval/mass;
 if(!std::isfinite(den)||!std::isfinite(r.residual)||!std::isfinite(r.mass_error)||r.residual>1e-10||r.mass_error>1e-10||!positive(candidate,mass)){r.outcome=Outcome::invariant_failed;return r;}
 r.state=candidate;r.has_state=true;r.outcome=Outcome::candidate;return r;
}
Result evaluate(const Matrix9& q,bool generator,double mass,Backend backend){
 if(!valid_input(q,mass))return Result{};
 return assess(q,backend(q),generator,mass);
}
const char* outcome_name(Outcome x){
 switch(x){
 #define NAME(n) case Outcome::n:return #n
 NAME(input_invalid);NAME(backend_failed);NAME(malformed_factors);NAME(svd_quality_failed);NAME(no_nullspace);NAME(nonunique);NAME(rank_ambiguous);NAME(normalization_unsafe);NAME(analytic_no_selection);NAME(invariant_failed);NAME(candidate);
 #undef NAME
 }return "unknown";
}
const char* normalization_name(Outcome x){
 switch(x){case Outcome::candidate:return "normalized_unique";case Outcome::nonunique:return "nonunique_no_selection";case Outcome::rank_ambiguous:return "rank_ambiguous_no_selection";case Outcome::normalization_unsafe:return "normalization_sum_unsafe";case Outcome::analytic_no_selection:return "analytic_no_physical_selection";default:return outcome_name(x);}
}
}

}
namespace checked {
namespace {
thread_local CallbackDiagnostic* capture=nullptr;
void bounded_copy(std::array<char,128>& out,const char* in,bool& truncated) noexcept {
 if(!in)return;
 size_t i=0;for(;i+1<out.size()&&in[i];++i)out[i]=in[i];
 out[i]=0;if(in[i])truncated=true;
}
void intercept(const char* reason,const char* file,int line,int code) noexcept {
 if(!capture)return; // Only the declared exclusive owner is supported.
 if(capture->count<INT_MAX)++capture->count;
 if(capture->count!=1)return;
 capture->code=code;capture->line=line;
 bounded_copy(capture->reason,reason,capture->truncated);
 bounded_copy(capture->file,file,capture->truncated);
}
struct Guard {
 gsl_error_handler_t* previous;
 explicit Guard(CallbackDiagnostic& d){capture=&d;previous=gsl_set_error_handler(&intercept);}
 ~Guard() noexcept {gsl_set_error_handler(previous);capture=nullptr;}
 Guard(const Guard&)=delete;Guard& operator=(const Guard&)=delete;
};
Status mapped(detail::Outcome o){return o==detail::Outcome::candidate?Status::success:static_cast<Status>(static_cast<int>(o)+1);}
bool enter(const Context& c,Report& r){
 if(!c.declared_exclusive()){r.status=Status::context_required;return false;}
 if(capture){r.status=Status::context_busy;return false;}
 return true;
}
}
#ifdef C_SPIKES_CHECKED_TESTING
namespace testing {
Fault& fault(){static thread_local Fault f;return f;}
void reset(){fault()=Fault{};}
}
#endif
const char* status_name(Status s){
 switch(s){
#define S(x) case Status::x:return #x
 S(success);S(input_invalid);S(backend_failed);S(malformed_factors);S(svd_quality_failed);
 S(no_nullspace);S(nonunique);S(rank_ambiguous);S(normalization_unsafe);S(analytic_no_selection);
 S(invariant_failed);S(context_required);S(context_busy);
#undef S
 }return "unknown";
}
struct Engine {
 static void params(GCaMP& m,CellParameters p){m.G_tot=p.G_tot;m.gamma=p.gamma;m.DCaT=p.DCaT;m.Rf=p.Rf;m.gam_in=p.gam_in;m.gam_out=p.gam_out;}
 static std::unique_ptr<GCaMP> empty(){
  auto m=std::unique_ptr<GCaMP>(new GCaMP(GCaMP::CheckedTag{}));
  m->state.zeros();m->initial_state.zeros();m->Gparams.zeros();m->Cparams.zeros();m->Gmat.zeros();m->brightStates={2,3,5,6,8};m->DFF=0;
  m->G0=m->Gsat=m->Ginit=0;
  return m;
 }
 static std::unique_ptr<GCaMP> stage(const GCaMP& m){
  auto s=empty();s->Gparams=m.Gparams;s->brightStates=m.brightStates;
#define COPY(x) s->x=m.x
  COPY(G_tot);COPY(gamma);COPY(DCaT);COPY(Rf);COPY(gam_in);COPY(gam_out);
  COPY(konN);COPY(koffN);COPY(konC);COPY(koffC);COPY(konPN);COPY(koffPN);COPY(konPN2);COPY(koffPN2);
  COPY(konPC);COPY(koffPC);COPY(konPC2);COPY(koffPC2);COPY(BCa0);COPY(kapB);COPY(sigma2_calcium_spike);
  COPY(G0);COPY(Gsat);COPY(Ginit);
#undef COPY
  return s;
 }
 static bool solve(GCaMP& m,double ca,Report& r,std::array<double,12>& out){
  ++r.solve_count;
#ifdef C_SPIKES_CHECKED_TESTING
  auto& fault=testing::fault();
  if(fault.nested&&r.solve_count==1)fault.nested();
  if(fault.call==r.solve_count){
   if(fault.unwind)throw std::runtime_error("test-only wrapper unwind");
   if(fault.callback_mode){
    gsl_error("forced callback diagnostic deliberately longer than the bounded buffer: abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz","g1e_test",123,GSL_EFAILED);
    r.backend_code=fault.callback_mode==2?0:GSL_EFAILED;r.status=Status::backend_failed;return false;
   }
   r.status=fault.status;return false;
  }
#endif
  m.setGmat(ca);detail::Matrix9 q{};
  for(size_t i=0;i<9;++i)for(size_t j=0;j<9;++j)q[9*i+j]=m.Gmat(i,j);
  auto s=detail::evaluate(q,true,m.G_tot);
  if(r.solve_count<=4)r.solves[r.solve_count-1]=s;
  r.backend_code=s.factors.code;
  if(r.callback.count){r.status=Status::backend_failed;return false;}
  r.status=mapped(s.outcome);if(!s.has_state)return false;
  for(size_t i=0;i<9;++i)out[i]=s.state[i];out[9]=m.BCa0;out[10]=out[11]=ca;
  return true;
 }
 static double bright(const GCaMP& m,const std::array<double,12>& x){double v=0;for(auto i:m.brightStates)v+=x[i];return v;}
 static bool norms(GCaMP& m,Report& r){
  std::array<double,12>a,b,c;
  if(!solve(m,0,r,a)||!solve(m,m.csat,r,b)||!solve(m,m.c0,r,c))return false;
  m.G0=bright(m,a);m.Gsat=bright(m,b);m.Ginit=bright(m,c);return true;
 }
 static void cache_commit(GCaMP& dst,const GCaMP& src){
  dst.konN=src.konN;dst.konC=src.konC;
  for(size_t i=0;i<81;++i)dst.Gmat[i]=src.Gmat[i];
 }
 static EquilibriumResult update(GCaMP& dst,Operation op,CellParameters cp,double ca,const Context& context){
  EquilibriumResult result;auto& r=result.report;r.operation=op;
  if(!enter(context,r))return result;
  Guard guard(r.callback);auto s=stage(dst);std::array<double,12> values{};
  const bool parameters=op==Operation::parameters||op==Operation::parameters_and_init;
  const bool reset=op==Operation::initialize||op==Operation::parameters_and_init;
  if(parameters){params(*s,cp);if(!norms(*s,r))return result;}
  if(op!=Operation::parameters){if(!solve(*s,reset?s->c0:ca,r,values))return result;}
  if(op==Operation::equilibrium)result.state=values; // Fixed storage, prepared before commit.
  if(parameters){
   params(dst,cp);dst.G0=s->G0;dst.Gsat=s->Gsat;dst.Ginit=s->Ginit;
  }
  cache_commit(dst,*s);
  if(op==Operation::state||reset)for(size_t i=0;i<12;++i){dst.state[i]=values[i];dst.initial_state[i]=values[i];}
  if(reset)dst.DFF=0;
  r.status=Status::success;r.committed=true;return result;
 }
 static CreateResult create(const std::array<double,14>& raw,CellParameters cp,const Context& context){
  CreateResult result;auto& r=result.report;r.operation=Operation::create;
  if(!enter(context,r))return result;
  Guard guard(r.callback);auto m=empty();params(*m,cp);
  for(size_t i=0;i<14;++i)m->Gparams[i]=raw[i];
  const std::array<double,6> c={cp.G_tot,cp.gamma,cp.DCaT,cp.Rf,cp.gam_in,cp.gam_out};
  for(size_t i=0;i<6;++i)m->Cparams[i]=c[i];
  m->Gparams(0)=m->Gparams(1)/pow(m->Gparams(0),m->Gparams(2));
  m->Gparams(3)=m->Gparams(4)/pow(m->Gparams(3),m->Gparams(5));
  m->kapB=m->B_tot/(m->koff_B/m->kon_B);
  m->BCa0=m->kon_B*m->c0/(m->kon_B*m->c0+m->koff_B)*m->B_tot;
  m->konN=m->Gparams(0)*pow(m->c0,m->Gparams(2));m->koffN=m->Gparams(1);
  m->konC=m->Gparams(3)*pow(m->c0,m->Gparams(5));m->koffC=m->Gparams(4);
  m->konPN=m->Gparams(6);m->koffPN=m->Gparams(7);m->konPC=m->Gparams(8);m->koffPC=m->Gparams(9);
  m->konPN2=m->Gparams(10);m->koffPN2=m->Gparams(11);m->konPC2=m->Gparams(12);m->koffPC2=m->Gparams(13);
  m->sigma2_calcium_spike=pow(m->FWHM,2)/(8*log(2));
  std::array<double,12> state;
  if(!solve(*m,m->c0,r,state))return result;
  for(size_t i=0;i<12;++i){m->state[i]=state[i];m->initial_state[i]=state[i];}
  if(!norms(*m,r))return result;
  // Factory evolution default only; equilibrium/staging are mode-independent.
  m->TSMode=FIXEDLA;
  result.model=std::move(m);r.status=Status::success;r.committed=true;return result;
 }
};
CreateResult try_create_gcamp(const std::array<double,14>& a,CellParameters p,const Context& c){return Engine::create(a,p,c);}
}
checked::UpdateResult GCaMP::try_set_params(checked::CellParameters p,const checked::Context& c){return {checked::Engine::update(*this,checked::Operation::parameters,p,0,c).report};}
checked::UpdateResult GCaMP::try_set_state(double ca,const checked::Context& c){return {checked::Engine::update(*this,checked::Operation::state,{},ca,c).report};}
checked::UpdateResult GCaMP::try_init(const checked::Context& c){return {checked::Engine::update(*this,checked::Operation::initialize,{},0,c).report};}
checked::UpdateResult GCaMP::try_set_params_and_init(checked::CellParameters p,const checked::Context& c){return {checked::Engine::update(*this,checked::Operation::parameters_and_init,p,0,c).report};}
checked::EquilibriumResult GCaMP::try_steady_state(double ca,const checked::Context& c){return checked::Engine::update(*this,checked::Operation::equilibrium,{},ca,c);}
#endif
