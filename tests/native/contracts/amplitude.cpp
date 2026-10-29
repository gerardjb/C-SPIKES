#include <armadillo>
#include <sstream>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <json/json.h>
#include <gsl/gsl_errno.h>
#include <type_traits>
#include <limits>
#include <iostream>
#include <cstring>
#include <cmath>
#include <memory>
csp_model::ScalarInputView input(const arma::vec&x){return {x.memptr(),x.n_elem};}
static_assert(std::is_constructible<GCaMP,std::string,std::string>::value,"filename constructor");
static_assert(std::is_constructible<GCaMP,double,double,double,double,double,double,std::string>::value,"scalar constructor");
#ifndef ARMA_CHECK_CONFORMANCE
#error baseline conformance must be enabled
#endif
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
std::string bits(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";const char*h="0123456789abcdef";unsigned char a[8];std::memcpy(a,&x,8);std::string s;for(auto c:a){s+=h[c>>4];s+=h[c&15];}return s;}
std::string hex(const std::string&s){const char*h="0123456789abcdef";std::string out;for(unsigned char c:s){out+=h[c>>4];out+=h[c&15];}return out;}
template<class V>Json::Value values(const V& v){Json::Value j(Json::arrayValue);for(auto x:v)j.append(bits(x));return j;}
Json::Value logs(Json::arrayValue);
void capture(const csp_diagnostics::Record&r){Json::Value j;j["operation"]=r.operation;j["method"]=r.first.method;j["field"]=r.first.field;j["mode"]=r.first.mode;j["index"]=Json::Int64(r.first.index);j["value"]=bits(r.first.value);j["count"]=Json::UInt64(r.detections);j["complete"]=r.complete;for(auto c:r.counts)j["counts"].append(Json::UInt64(c));logs.append(j);}
void handler(const char*,const char*,int,int){}
Json::Value observe(const GCaMP&m){Json::Value j;j["state"]=values(m.state);j["initial"]=values(m.initial_state);j["raw_cell"]=values(m.Cparams);j["scalars"]=values(checked::TestAccess::cell_scalars(m));j["indicator"]=values(m.Gparams);j["mode"]=int(m.TSMode);j["step"]=m.step_count;return j;}
std::unique_ptr<GCaMP> make(const arma::vec&gp,const arma::vec&cp){auto m=std::make_unique<GCaMP>(input(gp),input(cp));m->DFF=0.;return m;}
Json::Value construct(const arma::vec&gp,const arma::vec&cp){auto gb=values(gp),cb=values(cp);Json::Value j;logs=Json::Value(Json::arrayValue);try{csp_diagnostics::Scope scope("constructor-probe",capture);auto m=make(gp,cp);j["model"]=observe(*m);j["error"]="";need(m->TSMode==FIXED,"legacy mode");for(auto x:m->Cparams)need(x==0.,"raw Cparams synchronized");for(unsigned i=0;i<6;++i)need(bits(checked::TestAccess::cell_scalars(*m)[i])==bits(cp[i]),"cell mapping");need(bits(m->Gparams[0])==bits(gp[1]/std::pow(gp[0],gp[2]))&&bits(m->Gparams[3])==bits(gp[4]/std::pow(gp[3],gp[5])),"transformation changed");}
catch(const std::out_of_range&e){j["error"]=std::string("out_of_range:")+e.what();}catch(const std::logic_error&e){j["error"]=std::string("logic_error:")+e.what();}j["logs"]=logs;need(gb==values(gp)&&cb==values(cp),"input mutation");j["input_G"]=gb;j["input_C"]=cb;return j;}
int case_entry(int argc,char**argv){try{need(argc==4,"args");std::string id=argv[1];int k=std::stoi(id.substr(1));need(k>=1&&k<=8,"case");gsl_set_error_handler(handler);auto ctx=checked::Context::owned_isolated_process();arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");
 if(k==3||k==4)cp[2]=0.;if(k==5||k==6)cp[3]=7.;auto m=make(gp,cp);m->TSMode=(k%2)?FIXEDLA:FIXED;m->step_count=0;
 if(k==5||k==6)m->brightStates={2,2,5,5,8};if(k==7||k==8)checked::TestAccess::invalidate_normalization(*m);
 Json::Value out;out["id"]=id;logs=Json::Value(Json::arrayValue);out["before"]=observe(*m);
 {csp_diagnostics::Scope scope("amplitude-probe",capture);out["amplitude"]=bits(m->getAmplitude());}
 out["after"]=observe(*m);out["rates"]=values(checked::TestAccess::rates(*m));out["logs"]=logs;need(gsl_set_error_handler(handler)==handler,"handler restoration");out["ok"]=true;Json::StreamWriterBuilder writer;writer["indentation"]="";std::cout<<"RESULT "<<Json::writeString(writer,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){if(argc!=4)return 90;for(const char* id:{"T01","T02","T03","T04","T05","T06","T07","T08"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
