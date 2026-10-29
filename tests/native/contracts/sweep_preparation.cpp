#include "sweep_access.h"
#include <gsl/gsl_errno.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
void handler(const char*,const char*,int,int){}
void restored(){require(gsl_set_error_handler(&handler)==&handler,"handler not restored");}
template<class T>void raw(const T& v){std::cout<<'[';bool first=true;for(auto x:v){if(!first)std::cout<<',';first=false;std::cout<<x;}std::cout<<']';}
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
SMC* nesting=nullptr;param* nested_p=nullptr;bool busy_ok=false;
void nested(){auto m=checked::TestAccess::snapshot(*nesting->model);auto c=checked::SweepAccess::private_state(*nesting);auto r=nesting->try_prepare_sweep(*nested_p,checked::Context::owned_isolated_process());busy_ok=r.report.status==checked::Status::context_busy&&!r.report.committed&&!r.handoff&&r.report.solve_count==0&&m==checked::TestAccess::snapshot(*nesting->model)&&c==checked::SweepAccess::private_state(*nesting);}
#endif
int main(int argc,char**argv){try{
 require(argc==4,"arguments");std::string id=argv[1];std::cout<<std::setprecision(17);gsl_set_error_handler(&handler);
 constpar c(argv[3]);c.t_min=c.t_max=0;c.output_folder="unused-sentinel-output";c.tag="g1g";
 arma::vec time={0,.001,.002},data={0,0,0};SMC smc(time,data,0,c,false,17,0,argv[2]);auto& m=*smc.model;
 // Explicitly initialize only test snapshot fields that legacy construction leaves undefined.
 m.Cparams={11,12,13,14,15,16};checked::TestAccess::seed(m);
 for(auto row:smc.particleSystem)for(unsigned j=0;j<smc.nparticles;++j){row[j].C.fill(0.125);row[j].B=.25;row[j].burst=1;row[j].S=0;row[j].logWeight=-.5;row[j].ancestor=-1;}
 param p;p.G_tot=c.G_tot_mean;p.gamma=c.gamma_mean;p.DCaT=c.DCaT_mean;p.Rf=c.Rf_mean;p.gam_in=c.gam_in_mean;p.gam_out=c.gam_out_mean;p.wbb[0]=.1;p.wbb[1]=.2;p.r0=.3;p.r1=.4;p.sigma2=.5;p.filename="unused-param-output";
 Trajectory traj(3,"unused-trajectory-output");traj.B.fill(.1);traj.burst.fill(1);traj.C.fill(.2);traj.S.fill(0);traj.Y.fill(.3);
 if(id=="P02"||id=="P05"){p.G_tot=3e-5;p.gam_in=30;p.gam_out=10;}
 if(id=="P03")m.setState(0);if(id=="P04")m.setState(1e-6);if(id=="P05")m.setState(5e-8);
 m.setTimeStepMode(id=="P06"?FIXED:FIXEDLA);m.DFF=123;
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
 checked::SweepAccess::seed(smc);const auto private_before=checked::SweepAccess::private_state(smc);const auto caller_addresses=checked::SweepAccess::addresses(smc);
#endif
 const auto public_before=public_caller(smc,c,p,traj);const auto model_before=checked::TestAccess::snapshot(m);const auto model_addresses=checked::TestAccess::addresses(m);const auto history=checked::TestAccess::histories(m);const auto raw_cp=m.Cparams;
 auto context=checked::Context::owned_isolated_process();std::optional<GCaMP_params> h;std::vector<checked::detail::Result> trace;
 if(id[0]=='P'){
  for(int repeat=0;repeat<(id=="P08"?2:1);++repeat){
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
   auto r=smc.try_prepare_sweep(p,context);require(r.report.status==checked::Status::success&&r.report.committed&&r.handoff&&r.report.solve_count==4,"prepare success");h.emplace(*r.handoff);for(int i=0;i<r.report.solve_count;++i)trace.push_back(r.report.solves[i]);
#else
   m.setParams(p.G_tot,p.gamma,p.DCaT,p.Rf,p.gam_in,p.gam_out);m.init();h.emplace();m.read_params(*h);
#endif
  }
  GCaMP_params current;m.read_params(current);require(handoff_values(current)==handoff_values(*h),"named handoff fields");
  std::cout<<"RESULT {\"id\":\""<<id<<"\",\"state\":";raw(m.state);std::cout<<",\"initial\":";raw(m.initial_state);std::cout<<",\"selected\":";raw(m.state);
  std::cout<<",\"norm\":";raw(std::array<double,3>{h->G0,h->Gsat,h->Ginit});std::cout<<",\"params\":";raw(std::array<double,6>{h->G_tot,h->gamma,h->DCaT,h->Rf,h->gam_in,h->gam_out});
  std::cout<<",\"rates\":";raw(std::array<double,15>{h->konN,h->koffN,h->konC,h->koffC,h->konPN,h->koffPN,h->konPN2,h->koffPN2,h->konPC,h->koffPC,h->konPC2,h->koffPC2,h->BCa0,h->kapB,h->sigma2_calcium_spike});
  std::cout<<",\"Gparams\":";raw(m.Gparams);std::cout<<",\"matrix\":";raw(m.Gmat);std::cout<<",\"handoff\":";raw(handoff_values(*h));
  std::cout<<",\"mode\":"<<int(m.TSMode)<<",\"DFF_member\":"<<m.DFF<<",\"dff\":"<<m.getDFF()<<",\"selected_dff\":"<<m.getDFF()<<",\"trace\":[";
  for(size_t i=0;i<trace.size();++i){if(i)std::cout<<',';auto& x=trace[i];std::cout<<"{\"rank\":";raw(x.ranks);std::cout<<",\"residual\":"<<x.residual<<",\"mass_error\":"<<x.mass_error<<",\"minimum\":"<<x.minimum<<",\"selected\":"<<(x.has_state?"true":"false")<<'}';}
  std::cout<<"],\"ok\":true}"<<std::endl;
 }else{
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
  int n=std::stoi(id.substr(1));using checked::Status;
  if(n==1||n==2||n==3||n==6||n==7){
   auto& f=checked::testing::fault();f.call=n==1?2:n==2?3:n==7?1:4;f.status=n==2?Status::nonunique:Status::backend_failed;if(n==7)f.callback_mode=1;
   auto r=smc.try_prepare_sweep(p,context);require(r.report.status==f.status&&!r.report.committed&&!r.handoff&&r.report.solve_count==f.call,"forced failure report");
   require(model_before==checked::TestAccess::snapshot(m)&&model_addresses==checked::TestAccess::addresses(m),"model rollback");if(n==7)require(r.report.callback.count==1&&r.report.callback.code==GSL_EFAILED&&r.report.callback.truncated,"callback capture");
  }else if(n==4){
   auto r=smc.try_prepare_sweep(p,checked::Context{});require(r.report.status==Status::context_required&&!r.report.committed&&!r.handoff&&r.report.solve_count==0,"missing context");require(model_before==checked::TestAccess::snapshot(m)&&model_addresses==checked::TestAccess::addresses(m),"missing context rollback");restored();nesting=&smc;nested_p=&p;checked::testing::fault().nested=&nested;auto outer=smc.try_prepare_sweep(p,context);require(busy_ok&&outer.report.committed&&outer.handoff,"nested context");
  }else if(n==5){
   auto r=smc.try_prepare_sweep(p,context);require(r.report.status==Status::success&&r.report.committed&&r.handoff&&smc.model==model_addresses[0],"success identity");auto reacquired=checked::TestAccess::addresses(m);require(!reacquired.empty(),"reacquire");
  }else if(n==8){
   auto first=smc.try_prepare_sweep(p,context);require(first.report.committed&&first.handoff,"first snapshot");auto saved=handoff_values(*first.handoff);
   param other;other=p;other.wbb[0]=p.wbb[0];other.wbb[1]=p.wbb[1];other.G_tot=3e-5;other.gam_in=30;other.gam_out=10;
   auto second=smc.try_prepare_sweep(other,context);require(second.report.committed&&second.handoff&&saved==handoff_values(*first.handoff)&&saved!=handoff_values(*second.handoff),"snapshot independence");GCaMP_params current;m.read_params(current);require(handoff_values(current)==handoff_values(*second.handoff),"second named fields");
  }else throw std::runtime_error("unexpected contract");
  checked::testing::reset();std::cout<<"RESULT {\"id\":\""<<id<<"\",\"ok\":true}"<<std::endl;
#else
  throw std::runtime_error("baseline contracts not scheduled");
#endif
 }
 require(public_before==public_caller(smc,c,p,traj),"caller/config/parameters/trajectory mutation");require(smc.model==model_addresses[0],"model address");require(history==checked::TestAccess::histories(m)&&arma::all(raw_cp==m.Cparams),"history/Cparams drift");
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
 require(private_before==checked::SweepAccess::private_state(smc)&&caller_addresses==checked::SweepAccess::addresses(smc),"RNG/counter/private caller/storage mutation");
#endif
 restored();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
