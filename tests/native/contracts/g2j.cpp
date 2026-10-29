#include <armadillo>
#include <algorithm>
#include <sstream>
#include <memory>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <json/json.h>
#include <gsl/gsl_errno.h>
#include <limits>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <cstring>
void need(bool x,const char* s){if(!x)throw std::runtime_error(s);}
void handler(const char*,const char*,int,int){}
std::string hex(const std::string& s){const char* h="0123456789abcdef";std::string out;for(unsigned char c:s){out+=h[c>>4];out+=h[c&15];}return out;}
std::string bits(double v){return hex(std::string(reinterpret_cast<const char*>(&v),sizeof(v)));}
Json::Value indices(const GCaMP& m){Json::Value r(Json::arrayValue);for(auto x:m.brightStates)r.append(Json::UInt64(x));return r;}
Json::Value observe(GCaMP& m){Json::Value r;r["indices"]=indices(m);r["snapshot"]=hex(checked::TestAccess::snapshot(m));r["getDFF"]=bits(m.getDFF());arma::vec x(12);for(unsigned i=0;i<12;++i)x[i]=(i+1)*1e-8;r["supplied"]=bits(m.getDFF(csp_model::ScalarInputView{x.memptr(),x.n_elem}));return r;}
template<class F>std::string failure(F f){try{f();}catch(const std::out_of_range& e){return std::string("out_of_range:")+e.what();}throw std::runtime_error("expected bounds failure");}
int case_entry(int argc,char** argv){try{
 need(argc==4,"arguments");std::string id=argv[1];gsl_set_error_handler(&handler);auto ctx=checked::Context::owned_isolated_process();
 arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});m.setTimeStepMode(FIXEDLA);
 checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};Json::Value out;out["id"]=id;
 if(id=="I01"){
  static_assert(std::numeric_limits<arma::uword>::digits==64,"qualified index width");GCaMP copy(m);copy.brightStates[0]=1;need(m.brightStates[0]==2&&copy.brightStates[0]==1,"owned copy");out["model"]=observe(m);out["copy"]=indices(copy);
 }else if(id=="I06"){
  m.brightStates={9,10,11,9,10};out["model"]=observe(m);arma::vec short_state(9,arma::fill::zeros);out["bounds"]=failure([&]{m.getDFF(csp_model::ScalarInputView{short_state.memptr(),short_state.n_elem});});
 }else if(id=="I07"){
  for(auto bad:{std::uint64_t(12),std::numeric_limits<std::uint64_t>::max()}){
   m.brightStates={bad,3,5,6,8};const auto before=checked::TestAccess::snapshot(m);auto addresses=checked::TestAccess::addresses(m);
   out["errors"].append(failure([&]{m.getDFF();}));arma::vec v(12,arma::fill::zeros);out["errors"].append(failure([&]{m.getDFF(csp_model::ScalarInputView{v.memptr(),v.n_elem});}));need(before==checked::TestAccess::snapshot(m)&&addresses==checked::TestAccess::addresses(m),"failure changed model");
  }
 }else if(id=="I08"){
  m.brightStates={0,1,4,7,8};checked::TestAccess::seed(m);const auto before=checked::TestAccess::snapshot(m);const auto addresses=checked::TestAccess::addresses(m);checked::testing::reset();checked::testing::fault().call=1;checked::testing::fault().status=checked::Status::backend_failed;
  auto failed=m.try_set_params_and_init(cell,ctx);need(!failed.report.committed&&failed.report.status==checked::Status::backend_failed,"fault status");need(before==checked::TestAccess::snapshot(m)&&addresses==checked::TestAccess::addresses(m),"rollback/reference preservation");checked::testing::reset();auto result=m.try_set_params_and_init(cell,ctx);need(result.report.committed,"success");need(m.brightStates[0]==0&&m.brightStates[4]==8,"staged indices");GCaMP copy(m);copy.brightStates[1]=8;need(m.brightStates[1]==1,"copy independent");out["model"]=observe(m);
 }else{
  if(id=="I03")m.brightStates={8,6,5,3,2};else if(id=="I04")m.brightStates={2,2,5,5,8};else need(id=="I02"||id=="I05","case");
  m.setParams(cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]);out["before"]=observe(m);
  if(id=="I03"||id=="I04"){m.fixedStep_LA(1e-6,0);out["after"]=observe(m);need(std::isfinite(m.getDFF()),"finite control");}
  if(id=="I05"){
   m.brightStates={0,1,4,7,8};out["edited"]=observe(m);auto updated=m.try_set_params_and_init(cell,ctx);need(updated.report.committed,"checked prepare");out["prepared"]=observe(m);
  }
 }
 need(gsl_set_error_handler(&handler)==&handler,"handler restoration");out["ok"]=true;Json::StreamWriterBuilder b;b["indentation"]="";std::cout<<"RESULT "<<Json::writeString(b,out)<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"I01","I02","I03","I04","I05","I06","I07","I08"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
