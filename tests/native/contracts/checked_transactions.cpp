#include <armadillo>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <gsl/gsl_errno.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
using checked::Status;
void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
void outer(const char*,const char*,int,int){}
void handler_ok(){require(gsl_set_error_handler(&outer)==&outer,"handler not restored");}
void vec(const arma::vec& v){std::cout<<'[';for(size_t i=0;i<v.n_elem;++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
template<class T>void raw(const T& v){std::cout<<'[';bool first=true;for(auto x:v){if(!first)std::cout<<',';first=false;std::cout<<x;}std::cout<<']';}
void close(double x,double y){require(std::isfinite(x)&&std::isfinite(y)&&std::abs(x-y)<=1e-13+1e-10*std::abs(x),"numerical envelope");}
void parity(const GCaMP& a,const GCaMP& b){
 for(int i=0;i<12;++i){close(a.state[i],b.state[i]);close(a.initial_state[i],b.initial_state[i]);}
 GCaMP_params p,q;const_cast<GCaMP&>(a).read_params(p);const_cast<GCaMP&>(b).read_params(q);close(p.G0,q.G0);close(p.Gsat,q.Gsat);close(p.Ginit,q.Ginit);
 require(checked::TestAccess::nonnumeric(a)==checked::TestAccess::nonnumeric(b),"other member parity");
}
GCaMP* nested_model=nullptr;bool nested_ok=false;
void nested(){auto r=nested_model->try_init(checked::Context::owned_isolated_process());nested_ok=r.report.status==Status::context_busy&&!r.report.committed&&r.report.solve_count==0;}
int main(int argc,char**argv){try{
 require(argc==4,"arguments");const std::string id=argv[1];std::cout<<std::setprecision(17);gsl_set_error_handler(&outer);
 arma::vec gp,cp;require(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixture load");
 std::array<double,14> g{};std::copy(gp.begin(),gp.end(),g.begin());checked::CellParameters c{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]},changed{3e-5,1000,1e-5,5,30,10};auto context=checked::Context::owned_isolated_process();
 auto made=checked::try_create_gcamp(g,c,context);require(made.report.status==Status::success&&made.model,"factory");auto m=std::move(made.model);
 for(int i=0;i<6;++i)require(m->Cparams[i]==cp[i],"raw factory Cparams");
 std::vector<checked::detail::Result> trace;
 auto record=[&](const checked::Report& r){require(r.status==Status::success&&r.committed,"checked success");for(int i=0;i<r.solve_count;++i)trace.push_back(r.solves[i]);};record(made.report);
 {
  const int n=std::stoi(id.substr(1));auto* identity=m.get();
  require(m->try_set_state(1e-6,context).report.committed,"seed state");m->setTimeStepMode(FIXEDLA);checked::TestAccess::seed(*m);
  GCaMP expected(*m);const auto before=checked::TestAccess::snapshot(*m);const auto addresses=checked::TestAccess::addresses(*m);checked::Report r;
  if(n>=1&&n<=4){
   if(n==1){expected.setParams(changed.G_tot,changed.gamma,changed.DCaT,changed.Rf,changed.gam_in,changed.gam_out);r=m->try_set_params(changed,context).report;}
   if(n==2){expected.setState(0);r=m->try_set_state(0,context).report;}
   if(n==3){expected.init();r=m->try_init(context).report;}
   if(n==4){expected.setParams(changed.G_tot,changed.gamma,changed.DCaT,changed.Rf,changed.gam_in,changed.gam_out);expected.init();r=m->try_set_params_and_init(changed,context).report;}
   require(r.status==Status::success&&r.committed&&m.get()==identity,"success identity");parity(expected,*m);
  }else if(n>=5&&n<=10){
   auto& f=checked::testing::fault();f.call=n==5||n==9?2:n==6?3:n==8?4:1;f.status=n==6?Status::nonunique:n==7?Status::rank_ambiguous:n==10?Status::normalization_unsafe:Status::backend_failed;
   if(n==5||n==6)r=m->try_set_params(changed,context).report;
   if(n==7)r=m->try_set_state(0,context).report;
   if(n==8)r=m->try_set_params_and_init(changed,context).report;
   if(n==9){auto x=checked::try_create_gcamp(g,c,context);r=x.report;require(!x.model,"partial factory model");}
   if(n==10){auto x=m->try_steady_state(0,context);r=x.report;require(!x.state,"failed output candidate");}
   require(r.status==f.status&&!r.committed&&r.solve_count==f.call,"failure result");require(before==checked::TestAccess::snapshot(*m)&&addresses==checked::TestAccess::addresses(*m),"full rollback");
  }else if(n==11){
   const auto history=checked::TestAccess::histories(*m);require(m->try_set_params(changed,context).report.committed,"alias success");require(m.get()==identity&&history==checked::TestAccess::histories(*m),"history values");
   // Old storage handles are discarded without dereference; reacquire after success.
   auto current=checked::TestAccess::addresses(*m);auto bytes=checked::TestAccess::snapshot(*m);checked::testing::fault().call=4;
   r=m->try_set_params_and_init(c,context).report;require(!r.committed&&r.status==Status::backend_failed,"alias failure status");require(current==checked::TestAccess::addresses(*m)&&bytes==checked::TestAccess::snapshot(*m),"alias rollback");
  }else if(n==15){
   require(m->try_init(context).report.committed,"guard success");handler_ok();auto bytes=checked::TestAccess::snapshot(*m);
   for(int mode:{1,2}){checked::testing::reset();auto& f=checked::testing::fault();f.call=1;f.callback_mode=mode;r=m->try_init(context).report;require(r.status==Status::backend_failed&&!r.committed&&r.callback.count==1&&r.callback.truncated&&r.callback.code==GSL_EFAILED&&r.backend_code==(mode==2?0:GSL_EFAILED),"callback capture/return");require(bytes==checked::TestAccess::snapshot(*m),"callback rollback");handler_ok();}
   checked::testing::reset();checked::testing::fault().call=1;checked::testing::fault().unwind=true;bool caught=false;try{m->try_init(context);}catch(const std::runtime_error&){caught=true;}require(caught&&bytes==checked::TestAccess::snapshot(*m),"wrapper unwind");handler_ok();
  }else if(n==16){
   r=m->try_init(checked::Context{}).report;require(r.status==Status::context_required&&!r.committed&&r.solve_count==0&&before==checked::TestAccess::snapshot(*m),"missing context");handler_ok();nested_model=m.get();checked::testing::fault().nested=&nested;r=m->try_init(context).report;require(nested_ok&&r.committed,"nested rejection");
  }else throw std::runtime_error("unexpected native contract");
  checked::testing::reset();std::cout<<"RESULT {\"id\":\""<<id<<"\",\"ok\":true}"<<std::endl;
 }
 handler_ok();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}catch(...){std::cerr<<"unexpected exception"<<std::endl;return 3;}}
