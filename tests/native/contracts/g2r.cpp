#include <armadillo>
#include <algorithm>
#include <sstream>
#include <memory>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <json/json.h>
#include <gsl/gsl_errno.h>
#include <type_traits>
#include <limits>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cmath>
#ifdef G2R_BASELINE
const arma::vec& input(const arma::vec& x){return x;}
#else
csp_model::ScalarInputView input(const arma::vec& x){return {x.memptr(),x.n_elem};}
static_assert(std::is_same<decltype(&GCaMP::flux),double(GCaMP::*)(double,csp_model::ScalarInputView)>::value,"flux signature");
static_assert(std::is_same<decltype(&GCaMP::flux_konN_konC),double(GCaMP::*)(double,double,csp_model::ScalarInputView)>::value,"explicit rates signature");
static_assert(std::is_same<decltype(static_cast<double(GCaMP::*)(csp_model::ScalarInputView)>(&GCaMP::getDFF)),double(GCaMP::*)(csp_model::ScalarInputView)>::value,"DFF signature");
#endif
static_assert(sizeof(arma::uword)==sizeof(std::size_t)&&sizeof(arma::uword)==8,"qualified widths");
#ifndef ARMA_CHECK_CONFORMANCE
#error baseline conformance must be enabled
#endif
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
std::string bits(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";const char*h="0123456789abcdef";unsigned char a[8];std::memcpy(a,&x,8);std::string s;for(auto c:a){s+=h[c>>4];s+=h[c&15];}return s;}
template<class V>Json::Value values(const V& v){Json::Value j(Json::arrayValue);for(auto x:v)j.append(bits(x));return j;}
Json::Value logs(Json::arrayValue);
void capture(const csp_diagnostics::Record&r){Json::Value j;j["operation"]=r.operation;j["method"]=r.first.method;j["field"]=r.first.field;j["mode"]=r.first.mode;j["index"]=Json::Int64(r.first.index);j["value"]=bits(r.first.value);j["count"]=Json::UInt64(r.detections);j["complete"]=r.complete;for(auto c:r.counts)j["counts"].append(Json::UInt64(c));logs.append(j);}
void handler(const char*,const char*,int,int){}
template<class F>Json::Value call(GCaMP& m,F f){Json::Value j;auto before=checked::TestAccess::rates(m);logs=Json::Value(Json::arrayValue);try{csp_diagnostics::Scope scope("input-probe",capture);j["value"]=bits(f());j["error"]="";}catch(const std::out_of_range&e){j["error"]=std::string("out_of_range:")+e.what();}j["rates_before"]=values(before);j["rates_after"]=values(checked::TestAccess::rates(m));j["logs"]=logs;return j;}
arma::vec pattern(unsigned n){arma::vec v(n);for(unsigned i=0;i<n;++i)v[i]=(i+1)*1e-8;return v;}
Json::Value observe(GCaMP&m,const arma::vec& v,double ca=1e-7){Json::Value j;auto before=values(v);j["flux"]=call(m,[&]{return m.flux(ca,input(v));});auto rates=checked::TestAccess::rates(m);j["explicit"]=call(m,[&]{return m.flux_konN_konC(3.25,7.5,input(v));});need(rates==checked::TestAccess::rates(m),"explicit rates changed cache");j["DFF"]=call(m,[&]{return m.getDFF(input(v));});need(before==values(v),"input changed");j["input"]=before;return j;}
int case_entry(int argc,char**argv){try{need(argc==4,"args");std::string id=argv[1];gsl_set_error_handler(handler);auto ctx=checked::Context::owned_isolated_process();arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});m.setTimeStepMode(FIXEDLA);m.DFF=0.;Json::Value out;out["id"]=id;
 if(id=="N01"){out["nine"]=observe(m,pattern(9));out["twelve"]=observe(m,pattern(12));}
 else if(id=="N02"){auto v=pattern(15);auto a=observe(m,v);v[12]=-99.;v[13]=999.;v[14]=-1.;auto z=observe(m,v);need(a["flux"]["value"]==z["flux"]["value"]&&a["DFF"]["value"]==z["DFF"]["value"],"tail used");out["first"]=a;out["tail"]=z;}
 else if(id=="N03"){for(double ca:{0.,1e-7,1e-6})out["records"].append(observe(m,pattern(9),ca));}
 else if(id=="N04"){for(unsigned n=0;n<9;++n){auto row=observe(m,pattern(n),(n+1)*1e-8);need(row["flux"]["error"].asString()=="out_of_range:Mat::operator(): index out of bounds","flux bounds");need(row["DFF"]["error"].asString()=="out_of_range:Mat::elem(): index out of bounds","gather bounds");out["records"].append(row);}}
 else if(id=="N05"){m.brightStates={8,6,5,3,2};out["record"]=observe(m,pattern(12));}
 else if(id=="N06"){m.brightStates={2,2,5,5,8};out["record"]=observe(m,pattern(12));}
 else if(id=="N07"){m.brightStates={9,10,11,9,10};out["twelve"]=observe(m,pattern(12));m.brightStates={14,11,14,2,0};out["long"]=observe(m,pattern(15));}
 else if(id=="N08"){for(auto bad:{std::uint64_t(12),std::numeric_limits<std::uint64_t>::max()}){m.brightStates={bad,3,5,6,8};auto snap=checked::TestAccess::snapshot(m);auto v=pattern(12);auto j=call(m,[&]{return m.getDFF(input(v));});need(j["error"].asString()=="out_of_range:Mat::elem(): index out of bounds","bad gather");need(snap==checked::TestAccess::snapshot(m),"bad gather mutation");out["records"].append(j);}}
 else if(id=="N09"){auto v=pattern(12);const auto& view=input(v);out["first"]=call(m,[&]{return m.getDFF(view);});v[2]*=2;out["second"]=call(m,[&]{return m.getDFF(view);});need(out["first"]["value"]!=out["second"]["value"],"view copied input");
#ifndef G2R_BASELINE
 need(view.data()==v.memptr()&&view.size()==v.n_elem,"view identity");
#endif
 auto copied=v;v.reset();out["independent"]=observe(m,copied);}
 else if(id=="N10"){arma::vec s(m.state.data(),12,false,true),p(m.Gparams.data(),14,false,true);auto old_state=values(s),old_params=values(p);out["state"]=observe(m,s);out["parameters"]=observe(m,p);need(old_state==values(s)&&old_params==values(p),"aliased storage changed");}
 else if(id=="N11"){auto v=pattern(12);v[2]=std::numeric_limits<double>::quiet_NaN();out["nan"]=observe(m,v);v[2]=std::numeric_limits<double>::infinity();out["inf"]=observe(m,v);checked::TestAccess::invalidate_normalization(m);out["denominator"]=observe(m,pattern(12));}
 else if(id=="N12"){auto v=pattern(12);out["before"]=observe(m,v);auto snap=checked::TestAccess::snapshot(m);auto addr=checked::TestAccess::addresses(m);checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};checked::testing::reset();checked::testing::fault().call=1;checked::testing::fault().status=checked::Status::backend_failed;auto f=m.try_set_params_and_init(cell,ctx);need(!f.report.committed&&snap==checked::TestAccess::snapshot(m)&&addr==checked::TestAccess::addresses(m),"rollback");checked::testing::reset();auto good=m.try_set_params_and_init(cell,ctx);need(good.report.committed,"checked update");out["after"]=observe(m,v);need(values(v)==values(pattern(12)),"input lifetime");}
 else need(false,"case");need(gsl_set_error_handler(handler)==handler,"handler restoration");out["ok"]=true;Json::StreamWriterBuilder b;b["indentation"]="";std::cout<<"RESULT "<<Json::writeString(b,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"N01","N02","N03","N04","N05","N06","N07","N08","N09","N10","N11","N12"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
