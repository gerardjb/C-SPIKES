#include <armadillo>
#include <sstream>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <json/json.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <cstring>
#include <cmath>
#include <limits>
#include <functional>
void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
void handler(const char*,const char*,int,int){}
std::string bits(double v){if(std::isnan(v))return "nan";if(std::isinf(v))return v>0?"+inf":"-inf";const char*h="0123456789abcdef";unsigned char b[8];std::memcpy(b,&v,8);std::string s;for(auto c:b){s+=h[c>>4];s+=h[c&15];}return s;}
template<class V>Json::Value vals(const V&v){Json::Value j(Json::arrayValue);for(double x:v)j.append(bits(x));return j;}
Json::Value model(const GCaMP&m){Json::Value j;j["state"]=vals(m.state);j["initial"]=vals(m.initial_state);j["norms"]=vals(checked::TestAccess::norms(m));j["rates"]=vals(checked::TestAccess::rates(m));j["scalars"]=vals(checked::TestAccess::cell_scalars(m));j["Gmat"]=Json::Value(Json::arrayValue);for(unsigned i=0;i<81;++i)j["Gmat"].append(bits(m.Gmat.data()[i]));return j;}
Json::Value call(const std::function<double()>&f){Json::Value j;try{j["value"]=bits(f());j["status"]="ok";}catch(const std::out_of_range&e){j["status"]="out_of_range";j["message"]=e.what();need(std::string(e.what())=="Mat::elem(): index out of bounds","bounds text");}return j;}
int case_entry(int argc,char**argv){try{
 need(argc==4,"args");std::string id=argv[1];arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");gsl_set_error_handler(&handler);GCaMP m({gp.memptr(),gp.n_elem},{cp.memptr(),cp.n_elem});m.setTimeStepMode(FIXEDLA);Json::Value out;out["id"]=id;
 auto both=[&](){Json::Value j;const auto before=model(m);j["member"]=call([&](){return m.getDFF();});j["view"]=call([&](){return m.getDFF({m.state.data(),12});});need(j["member"]==j["view"],"two getters agreement");need(before==model(m),"getter mutation");return j;};
 if(id=="B01"){out["values"]=both();}
 else if(id=="B02"){m.brightStates={8,6,5,3,2};out["reversed"]=both();m.brightStates={2,2,2,2,2};out["duplicates"]=both();}
 else if(id=="B03"){out["before"]=both();m.brightStates={0,1,4,7,8};m.state[0]=.125;out["after"]=both();need(out["before"]!=out["after"],"member edits seen");}
 else if(id=="B04"||id=="B05"){for(unsigned i=(id=="B05"?4:0);i<(id=="B05"?5:4);++i){m.brightStates={0,1,2,3,4};m.brightStates[i]=12;auto j=both();need(j["member"]["status"]=="out_of_range","invalid index failure");out["cases"].append(j);}}
 else if(id=="B06"){double s[15];for(unsigned i=0;i<15;++i)s[i]=.125*i;m.brightStates={0,0,0,0,0};out["one"]=call([&](){return m.getDFF({s,1});});out["empty"]=call([&](){return m.getDFF({nullptr,0});});need(out["empty"]["status"]=="out_of_range","empty error");m.brightStates={14,13,12,11,10};out["long"]=call([&](){return m.getDFF({s,15});});out["short"]=call([&](){return m.getDFF({s,14});});need(out["short"]["status"]=="out_of_range","actual length error");}
 else if(id=="B07"){m.brightStates={0,1,2,3,4};checked::TestAccess::norms_for_test(m,0.,2.,0.,2.);for(auto v:{std::array<double,5>{1e16,1.,-1e16,1.,1.},{-0.,-0.,-0.,-0.,-0.}}){m.state.fill(0.);std::copy(v.begin(),v.end(),m.state.begin());out["cases"].append(both());}}
 else if(id=="B08"){m.brightStates={0,1,2,3,4};m.state.fill(0.);checked::TestAccess::norms_for_test(m,0.,2.,0.,2.);m.state[0]=std::numeric_limits<double>::infinity();out["inf"]=both();m.state[2]=-std::numeric_limits<double>::infinity();out["nan"]=both();m.state.fill(0.);checked::TestAccess::invalidate_normalization(m);out["zero_denominator"]=both();}
 else if(id=="B09"){out["setup"]=model(m);m.setParams(cp[0]*1.125,cp[1],cp[2],cp[3],cp[4],cp[5]);out["valid"]=model(m);const auto norms=checked::TestAccess::norms(m);m.brightStates[4]=12;out["error"]=call([&](){m.setParams(cp[0]*1.25,cp[1],cp[2],cp[3],cp[4],cp[5]);return 0.;});need(out["error"]["status"]=="out_of_range","normalization invalid-index error");need(checked::TestAccess::cell_scalars(m)[0]==cp[0]*1.25,"scalars mutated before error");need(norms==checked::TestAccess::norms(m),"no first norm publication on failed gather");out["partial"]=model(m);}
 else if(id=="B10"){checked::TestAccess::seed(m);const auto h=checked::TestAccess::histories(m);const auto ha=checked::TestAccess::history_addresses(m);m.setState(1e-6);need(std::equal(m.state.begin(),m.state.end(),m.initial_state.begin()),"state/initial publication");need(h==checked::TestAccess::histories(m)&&ha==checked::TestAccess::history_addresses(m),"histories preserved");out["legacy"]=model(m);auto before=checked::TestAccess::snapshot(m);auto refs=checked::TestAccess::addresses(m);checked::testing::reset();checked::testing::fault().call=2;checked::testing::fault().status=checked::Status::backend_failed;auto r=m.try_set_params_and_init({cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]},checked::Context::owned_isolated_process());checked::testing::reset();need(!r.report.committed&&r.report.status==checked::Status::backend_failed,"checked failure");need(before==checked::TestAccess::snapshot(m)&&refs==checked::TestAccess::addresses(m),"checked rollback");out["rollback"]=true;}
 else if(id=="B11"||id=="B12"){m.setTimeStepMode(id=="B11"?FIXEDLA:FIXED);m.setState(1e-6);m.brightStates={8,6,5,3,2};m.evolve(1e-4,0);out["after"]=model(m);out["dff"]=both();m.brightStates={8,6,5,3,9};out["error"]=call([&](){m.evolve(1e-4,0);return 0.;});need(out["error"]["status"]=="out_of_range","nine-state bound");out["partial"]=model(m);}
 else throw std::runtime_error("case");
 need(gsl_set_error_handler(&handler)==&handler,"handler unchanged");out["ok"]=true;Json::StreamWriterBuilder b;b["indentation"]="";std::cout<<"RESULT "<<Json::writeString(b,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){if(argc!=4)return 90;for(const char* id:{"B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
