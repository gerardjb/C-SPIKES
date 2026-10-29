#ifdef C_SPIKES_CHECKED_CPU
#include "include/GCaMP_model.h"
#include "include/checked_equilibrium.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <climits>
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
  m->state.fill(0.);m->initial_state.fill(0.);m->Gparams.fill(0.);m->Cparams.fill(0.);m->Gmat.fill(0.);m->brightStates={2,3,5,6,8};m->DFF=0;
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
