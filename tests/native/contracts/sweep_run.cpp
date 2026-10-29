#include "sweep_access.h"
#include "trajectory_test_compat.h"
#include <json/json.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
void handler(const char*,const char*,int,int){}
void restored(){require(gsl_set_error_handler(&handler)==&handler,"handler not restored");}
template<class T> Json::Value json_array(const T& v){Json::Value j(Json::arrayValue);for(auto x:v){require(std::isfinite(double(x)),"nonfinite output");j.append(double(x));}return j;}
std::string hex(const void* ptr,size_t n){const auto* p=static_cast<const unsigned char*>(ptr);const char* h="0123456789abcdef";std::string s;for(size_t i=0;i<n;++i){s+=h[p[i]>>4];s+=h[p[i]&15];}return s;}
std::string rng_state(const SMC& s){auto a=checked::SweepAccess::addresses(s);auto* r=static_cast<const gsl_rng*>(a[3]);return hex(gsl_rng_state(r),gsl_rng_size(r));}
uint64_t counter(const SMC& s){auto bytes=checked::SweepAccess::private_state(s);require(bytes.size()>=sizeof(uint64_t),"counter snapshot");uint64_t n;std::memcpy(&n,bytes.data(),sizeof(n));return n;}
std::string caller_inputs(const constpar& c,const param& p,const Trajectory& t){return config_bytes(c)+param_bytes(p)+trajectory_bytes(t);}
std::vector<const void*> trajectory_addresses(const Trajectory& t){return {&t,pointer(t.B),pointer(t.burst),pointer(t.C),pointer(t.S),pointer(t.Y)};}
#ifdef C_SPIKES_CHECKED_SWEEP_RUN_CPU
SMC* nested_s=nullptr;param* nested_p=nullptr;Trajectory* nested_in=nullptr;Trajectory* nested_out=nullptr;bool busy_ok=false;
void nested(){auto m=checked::TestAccess::snapshot(*nested_s->model);auto s=checked::SweepAccess::private_state(*nested_s);auto r=nested_s->try_PGAS(*nested_p,*nested_in,*nested_out,checked::Context::owned_isolated_process());busy_ok=r.preparation.status==checked::Status::context_busy&&!r.preparation.committed&&!r.sweep_returned&&r.preparation.solve_count==0&&m==checked::TestAccess::snapshot(*nested_s->model)&&s==checked::SweepAccess::private_state(*nested_s);}
#endif
int run(int argc,char**argv){
 require(argc==4,"arguments");std::string id=argv[1];bool success=id[0]=='P';int number=std::stoi(id.substr(1));int family=success?(number-1)/2:0;int seed=success&&number%2==0?23:17;
 gsl_set_error_handler(&handler);constpar c(argv[3]);c.t_min=c.t_max=0;c.seed=seed;c.output_folder="unused-output";c.tag="g1h";
 csp_application::ObservationVector time={0,.001,.002},data={0,0,0};SMC smc(time,data,0,c,false,seed,0,argv[2]);auto& m=*smc.model;
 m.Cparams.coefficients={11,12,13,14,15,16};checked::TestAccess::seed(m);checked::SweepAccess::seed(smc);m.setTimeStepMode(FIXEDLA);m.DFF=123;
 require(smc.nparticles==2&&smc.TIME==3,"frozen dimensions");
 for(auto row:smc.particleSystem)for(unsigned j=0;j<smc.nparticles;++j){fill_values(row[j].C,.125);row[j].B=.25;row[j].burst=1;row[j].S=0;row[j].logWeight=-.5;row[j].ancestor=-1;}
 param p;p.G_tot=c.G_tot_mean;p.gamma=c.gamma_mean;p.DCaT=c.DCaT_mean;p.Rf=c.Rf_mean;p.gam_in=c.gam_in_mean;p.gam_out=c.gam_out_mean;p.wbb[0]=.1;p.wbb[1]=.2;p.r0=.3;p.r1=.4;p.sigma2=.5;p.filename="unused-param-output";
 if(family==1||family==3){p.G_tot=3e-5;p.gam_in=30;p.gam_out=10;}
 Trajectory input(3,"unused-input"),output(3,"unused-output");fill_values(input.B,.1);fill_values(input.burst,1);fill_values(input.C,.2);fill_values(input.S,0);fill_values(input.Y,.3);if(family>=2)value_at(input.S,1)=1;
 fill_values(output.B,-7);fill_values(output.burst,1);fill_values(output.C,-9);fill_values(output.S,0);fill_values(output.Y,-8);
 auto inputs=caller_inputs(c,p,input);auto out_before=trajectory_bytes(output);auto public_before=public_caller(smc,c,p,input);auto private_before=checked::SweepAccess::private_state(smc);auto model_before=checked::TestAccess::snapshot(m);auto model_refs=checked::TestAccess::addresses(m);auto caller_refs=checked::SweepAccess::addresses(smc);auto out_refs=trajectory_addresses(output);auto rng_before=rng_state(smc);const auto before=counter(smc);auto histories=checked::TestAccess::histories(m);auto raw_cp=m.Cparams;
 Json::Value result;result["id"]=id;result["trace"]=Json::Value(Json::arrayValue);
 if(success){
#ifdef C_SPIKES_CHECKED_SWEEP_RUN_CPU
  auto r=smc.try_PGAS(p,input,output,checked::Context::owned_isolated_process());require(r.preparation.status==checked::Status::success&&r.preparation.committed&&r.preparation.solve_count==4&&r.sweep_returned,"sweep return status");
  for(int i=0;i<4;++i){const auto& x=r.preparation.solves[i];Json::Value j;j["rank"]=json_array(x.ranks);j["residual"]=x.residual;j["mass_error"]=x.mass_error;j["minimum"]=x.minimum;j["selected"]=x.has_state;result["trace"].append(j);}
#else
  smc.PGAS(p,input,output);
#endif
  require(counter(smc)==before+1&&rng_state(smc)!=rng_before,"one sweep advancement");
  require(inputs==caller_inputs(c,p,input),"input/config mutation");require(out_refs==trajectory_addresses(output)&&output.size==3&&output.filename=="unused-output","output storage/metadata");
  require(pointer(output.C)!=pointer(input.C),"output aliases input");
  result["rng"]=rng_state(smc);result["counter"]=Json::UInt64(counter(smc));result["state"]=json_array(m.state);result["initial"]=json_array(m.initial_state);result["Gparams"]=json_array(m.Gparams);result["matrix"]=json_array(m.Gmat);result["mode"]=int(m.TSMode);
  GCaMP_params cp;m.read_params(cp);result["handoff"]=json_array(handoff_values(cp));result["params"]=json_array(std::array<double,6>{cp.G_tot,cp.gamma,cp.DCaT,cp.Rf,cp.gam_in,cp.gam_out});result["norm"]=json_array(std::array<double,3>{cp.G0,cp.Gsat,cp.Ginit});result["dff"]=m.getDFF();require(std::isfinite(m.getDFF()),"model DFF nonfinite");
  std::vector<double> states,baselines,discrete,weights;
  for(unsigned t=0;t<smc.TIME;++t)for(unsigned i=0;i<smc.nparticles;++i){auto& x=smc.particleSystem[t][i];for(auto z:x.C)states.push_back(z);baselines.push_back(x.B);discrete.insert(discrete.end(),{double(x.burst),double(x.S),double(x.ancestor),double(x.index)});weights.push_back(x.logWeight);}
  result["cloud_state"]=json_array(states);result["cloud_B"]=json_array(baselines);result["cloud_discrete"]=json_array(discrete);result["log_weights_diagnostic"]=json_array(weights);
  result["output_B"]=json_array(output.B);result["output_burst"]=json_array(output.burst);result["output_S"]=json_array(output.S);result["output_Y"]=json_array(output.Y);result["output_DFF"]=json_array(output.C);
  int selected=-1;for(unsigned i=0;i<smc.nparticles;++i)if(smc.particleSystem[2][i].B==value_at(output.B,2)){require(selected==-1,"ambiguous selected-particle observer");selected=int(i);}require(selected>=0,"selected particle absent");
  Json::Value selected_states(Json::arrayValue),path(Json::arrayValue);selected_states.resize(3);path.resize(3);
  for(int t=2;t>=0;--t){require(selected>=0&&selected<int(smc.nparticles),"invalid ancestor");const auto& x=smc.particleSystem[t][selected];require(x.B==value_at(output.B,t)&&x.burst==value_at(output.burst,t)&&x.S==value_at(output.S,t),"selected observer mismatch");selected_states[t]=json_array(x.C);path[t]=selected;if(t)selected=x.ancestor;}
  result["selected_states"]=selected_states;result["selected_path"]=path;
 }else{
#ifdef C_SPIKES_CHECKED_SWEEP_RUN_CPU
  using checked::Status;auto& f=checked::testing::fault();
  if(number==7){
   nested_s=&smc;nested_p=&p;nested_in=&input;nested_out=&output;f.call=1;f.status=Status::backend_failed;f.nested=&nested;
   auto outer=m.try_set_params_and_init({p.G_tot,p.gamma,p.DCaT,p.Rf,p.gam_in,p.gam_out},checked::Context::owned_isolated_process());require(busy_ok&&!outer.report.committed&&outer.report.status==Status::backend_failed,"busy context isolation");
  }else{
   auto context=number==6?checked::Context{}:checked::Context::owned_isolated_process();Status expected=number==6?Status::context_required:number==5?Status::nonunique:Status::backend_failed;
   if(number<=4)f.call=number;else if(number==5)f.call=3;else if(number==8){f.call=1;f.callback_mode=1;}f.status=expected;
   auto r=smc.try_PGAS(p,input,output,context);require(r.preparation.status==expected&&!r.preparation.committed&&!r.sweep_returned&&r.preparation.solve_count==f.call,"failure propagation");
   if(number==8)require(r.preparation.callback.count==1&&r.preparation.callback.code==GSL_EFAILED&&r.preparation.callback.truncated,"bounded callback");
  }
  checked::testing::reset();require(model_before==checked::TestAccess::snapshot(m)&&model_refs==checked::TestAccess::addresses(m),"model rollback/references");require(private_before==checked::SweepAccess::private_state(smc)&&public_before==public_caller(smc,c,p,input)&&out_before==trajectory_bytes(output),"failure caller/output mutation");require(rng_before==rng_state(smc)&&counter(smc)==before,"failure advancement");
#else
  throw std::runtime_error("baseline failure cases are not scheduled");
#endif
 }
 require(inputs==caller_inputs(c,p,input)&&caller_refs==checked::SweepAccess::addresses(smc)&&out_refs==trajectory_addresses(output),"caller identity/input/storage");require(histories==checked::TestAccess::histories(m)&&std::equal(raw_cp.begin(),raw_cp.end(),m.Cparams.begin()),"history/Cparams mutation");restored();result["ok"]=true;
 Json::StreamWriterBuilder writer;writer["indentation"]="";writer["precision"]=17;std::cout<<"RESULT "<<Json::writeString(writer,result)<<std::endl;return 0;
}
int main(int argc,char**argv){int code=0;try{code=run(argc,argv);}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;code=2;}if(Kokkos::is_initialized())Kokkos::finalize();return code;}
