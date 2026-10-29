#include <armadillo>
#include <algorithm>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include "numerical_diagnostics.h"
#include <gsl/gsl_errno.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace csp_diagnostics;
static Record last;
static unsigned records=0;
static void capture(const Record& r){last=r;++records;}
static void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
static void handler(const char*,const char*,int,int){}
static void restored(){need(gsl_set_error_handler(&handler)==&handler,"handler restoration");}
static void na(const Location& p){need(p.mode==-1&&std::strcmp(p.integrator,"not_applicable")==0&&!p.timing_known&&p.requested_dt==0&&p.h==0&&p.substep==-1,"matrix metadata not N/A");}
static GCaMP_params params(GCaMP& m){GCaMP_params p{};m.read_params(p);return p;}
static bool same(const csp_model::State12& a,const csp_model::State12& b){return !std::memcmp(a.data(),b.data(),12*sizeof(double));}
int main(int argc,char**argv){try{
 need(argc==4,"args");std::string id=argv[1];arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii)&&gp.n_elem==14&&cp.n_elem==6,"fixture");
 gsl_set_error_handler(&handler);auto ctx=checked::Context::owned_isolated_process();std::array<double,14> raw{};std::copy(gp.begin(),gp.end(),raw.begin());checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};
 auto make=[&](){auto r=checked::try_create_gcamp(raw,cell,ctx);need(r.model&&r.report.committed&&r.report.status==checked::Status::success,"factory");return std::move(r.model);};
 const double bad=std::numeric_limits<double>::quiet_NaN();
 if(id=="D01"){
  {Scope scope("finite-context",capture);auto m=make();for(auto mode:{FIXED,FIXEDLA}){m->setTimeStepMode(mode);auto state=m->state,initial=m->initial_state;csp_model::Generator9 q;m->setGmat(0);m->fillGmat(q,GCaMP_params{}.c0);m->setGmat_konN_konC(q,params(*m).konN,params(*m).konC);auto equilibrium=m->steady_state(GCaMP_params{}.c0);need(std::all_of(equilibrium.begin(),equilibrium.end(),[](double v){return std::isfinite(v);})&&same(state,m->state)&&same(initial,m->initial_state)&&m->TSMode==mode,"finite context changed model state/mode");}}
  need(records==0,"finite warning");
 }else if(id=="D02"){
  const char* names[]={"GCaMP.setGmat","GCaMP.fillGmat","GCaMP.setGmat_konN_konC"};
  for(int i=0;i<3;++i){auto m=make();unsigned before=records;{Scope scope("invalid-generator",capture);scope.step(1,"FIXEDLA",.001,.0001,7,1e-6,2e-6,3e-6);csp_model::Generator9 q;if(i<2){m->Gparams(0)=bad;if(i==0)m->setGmat(GCaMP_params{}.c0);else m->fillGmat(q,GCaMP_params{}.c0);}else m->setGmat_konN_konC(q,bad,params(*m).konC);}
   need(records==before+1&&last.detections==1&&last.counts[rate_value]==1&&std::strcmp(last.first.method,names[i])==0&&std::strcmp(last.first.field,"generator")==0,"generator observation count/location");na(last.first);}
 }else if(id=="D03"){
  {Scope s("GCaMP.steady_state",capture);s.mode(1);s.equilibrium_context();s.scalar(bad,"steady_state",state_value,2);}
  need(records==1&&last.detections==1&&last.counts[state_value]==1&&last.first.index==2&&std::strcmp(last.first.field,"steady_state")==0,"equilibrium observation");na(last.first);
 }else if(id=="D04"){
  {Scope outer("existing-evolution",capture);outer.step(1,"FIXEDLA",.002,.0001,19,1e-6,2e-6,3e-6);
   {Scope inner("matrix");inner.equilibrium_context();na(active()->record.current);need(active()->record.current.calcium_known&&active()->record.current.ca==1e-6,"available calcium lost");inner.scalar(bad,"generator",rate_value);}
   auto p=active()->record.current;need(p.mode==1&&std::strcmp(p.integrator,"FIXEDLA")==0&&p.requested_dt==.002&&p.h==.0001&&p.substep==19&&p.timing_known&&p.calcium_known&&p.ca==1e-6&&p.bca==2e-6&&p.ca_in==3e-6,"outer context not restored");outer.scalar(bad,"state",state_value);}
  need(records==1&&last.detections==2,"aggregation changed");na(last.first);char a[1024],b[768],c[512];format(last,a,sizeof(a),b,sizeof(b),c,sizeof(c));need(std::strstr(a,"integrator=not_applicable")&&std::strstr(b,"suppressed=1"),"formatted metadata");
 }else if(id=="D05"){
  {Scope s("checked-valid",capture);auto m=make();need(m->TSMode==FIXEDLA,"factory default");for(auto mode:{FIXED,FIXEDLA}){m->setTimeStepMode(mode);auto check=[&](const checked::Report& r){need(r.committed&&r.status==checked::Status::success&&m->TSMode==mode,"checked mutator mode/status");restored();};check(m->try_set_params(cell,ctx).report);check(m->try_set_state(1e-6,ctx).report);check(m->try_init(ctx).report);check(m->try_set_params_and_init(cell,ctx).report);auto r=m->try_steady_state(0,ctx);check(r.report);need(bool(r.state),"missing owned equilibrium");}}
  need(records==0,"finite checked warning");
 }else if(id=="D06"){
  auto m=make();checked::TestAccess::seed(*m);auto before=checked::TestAccess::snapshot(*m);auto addresses=checked::TestAccess::addresses(*m);
  {Scope s("checked-failure",capture);checked::testing::reset();auto& f=checked::testing::fault();f.call=1;f.status=checked::Status::backend_failed;auto r=m->try_steady_state(GCaMP_params{}.c0,ctx);checked::testing::reset();need(!r.report.committed&&!r.state&&r.report.status==checked::Status::backend_failed,"failed result publication");}
  need(records==0,"finite backend status spuriously warned");need(before==checked::TestAccess::snapshot(*m)&&addresses==checked::TestAccess::addresses(*m),"failed update changed state/references");
 }else throw std::runtime_error("case");
 restored();std::cout<<"RESULT {\"id\":\""<<id<<"\",\"ok\":true,\"captured_records\":"<<records<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
