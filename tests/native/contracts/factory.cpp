#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <gsl/gsl_errno.h>
#include <json/json.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
using checked::Status;
void need(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
void handler(const char*,const char*,int,int){}
void restored(){need(gsl_set_error_handler(&handler)==&handler,"handler restoration");}
std::string hex(const std::string& s){const char* h="0123456789abcdef";std::string out;for(unsigned char c:s){out+=h[c>>4];out+=h[c&15];}return out;}
template<class A>Json::Value values(const A& a){Json::Value j(Json::arrayValue);for(auto x:a){need(std::isfinite(double(x)),"nonfinite observation");j.append(double(x));}return j;}
Json::Value observe(const GCaMP& m){
 Json::Value j;j["mode"]=int(m.TSMode);j["state"]=values(m.state);j["initial"]=values(m.initial_state);j["Cparams"]=values(m.Cparams);
 GCaMP copy(m);copy.setTimeStepMode(FIXED);j["mode_neutral_named_snapshot"]=hex(checked::TestAccess::snapshot(copy));return j;
}
int main(int argc,char**argv){try{
 need(argc==4,"arguments");std::string id=argv[1];gsl_set_error_handler(&handler);auto ctx=checked::Context::owned_isolated_process();
 arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");need(gp.n_elem==14&&cp.n_elem==6,"dimensions");std::array<double,14> raw{};std::copy(gp.begin(),gp.end(),raw.begin());
 checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};if(id=="P02")cell={3e-5,1000,1e-5,5,30,10};
 Json::Value out;out["id"]=id;
 if(id=="P04"){
  GCaMP a(gp,cp),b(cp[0],cp[1],cp[2],cp[3],cp[4],cp[5],argv[2]),c(std::string{argv[2]},std::string{argv[3]});
  for(auto* m:{&a,&b,&c}){m->Cparams=cp;need(m->TSMode==FIXED,"legacy default changed");out["legacy"].append(observe(*m));}
 }else if(id=="C02"){
  const auto raw_before=raw;const auto cp_before=values(cp);
  for(int call=1;call<=4;++call){checked::testing::reset();auto& f=checked::testing::fault();f.call=call;f.status=Status::backend_failed;auto r=checked::try_create_gcamp(raw,cell,ctx);need(!r.model&&!r.report.committed&&r.report.status==Status::backend_failed&&r.report.solve_count==call,"failed factory publication");restored();}
  checked::testing::reset();auto r=checked::try_create_gcamp(raw,cell,checked::Context{});need(!r.model&&!r.report.committed&&r.report.status==Status::context_required&&r.report.solve_count==0,"context failure");need(raw==raw_before&&values(cp)==cp_before,"input mutation");
 }else{
  auto made=checked::try_create_gcamp(raw,cell,ctx);need(made.model&&made.report.committed&&made.report.status==Status::success&&made.report.solve_count==4,"factory success");auto& m=*made.model;
  if(id=="P03")need(m.try_set_state(1e-6,ctx).report.committed,"state update");
  else if(id=="C01"){
   need(m.TSMode==FIXEDLA,"candidate factory mode");
   for(auto mode:{FIXED,FIXEDLA}){
    m.setTimeStepMode(mode);
    auto check=[&](const checked::Report& r){need(r.committed&&r.status==Status::success&&m.TSMode==mode,"mutator mode preservation");restored();};
    check(m.try_set_params(cell,ctx).report);check(m.try_set_state(1e-6,ctx).report);check(m.try_init(ctx).report);check(m.try_set_params_and_init(cell,ctx).report);check(m.try_steady_state(0,ctx).report);
   }
  }else need(id=="P01"||id=="P02","unknown case");
  out["model"]=observe(m);
 }
 restored();out["ok"]=true;Json::StreamWriterBuilder b;b["indentation"]="";b["precision"]=17;std::cout<<"RESULT "<<Json::writeString(b,out)<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
