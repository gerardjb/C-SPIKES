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
#include <cstring>
#include <cmath>
#include <memory>
#ifdef G2W_BASELINE
const arma::vec& input(const arma::vec&x){return x;}
#else
csp_model::ScalarInputView input(const arma::vec&x){return {x.memptr(),x.n_elem};}
static_assert(std::is_constructible<GCaMP,csp_model::ScalarInputView,csp_model::ScalarInputView>::value,"new constructor");
static_assert(!std::is_constructible<GCaMP,arma::vec,arma::vec>::value,"old constructor absent");
#endif
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
Json::Value observe(const GCaMP&m){Json::Value j;j["state"]=values(m.state);j["initial"]=values(m.initial_state);j["raw_cell"]=values(m.Cparams);j["scalars"]=values(checked::TestAccess::cell_scalars(m));j["indicator"]=values(m.Gparams);j["mode"]=int(m.TSMode);j["step"]=m.step_count;j["full_initialized_snapshot"]=hex(checked::TestAccess::snapshot(m));return j;}
std::unique_ptr<GCaMP> make(const arma::vec&gp,const arma::vec&cp){auto m=std::make_unique<GCaMP>(input(gp),input(cp));m->DFF=0.;return m;}
Json::Value construct(const arma::vec&gp,const arma::vec&cp){auto gb=values(gp),cb=values(cp);Json::Value j;logs=Json::Value(Json::arrayValue);try{csp_diagnostics::Scope scope("constructor-probe",capture);auto m=make(gp,cp);j["model"]=observe(*m);j["error"]="";need(m->TSMode==FIXED,"legacy mode");for(auto x:m->Cparams)need(x==0.,"raw Cparams synchronized");for(unsigned i=0;i<6;++i)need(bits(checked::TestAccess::cell_scalars(*m)[i])==bits(cp[i]),"cell mapping");need(bits(m->Gparams[0])==bits(gp[1]/std::pow(gp[0],gp[2]))&&bits(m->Gparams[3])==bits(gp[4]/std::pow(gp[3],gp[5])),"transformation changed");}
catch(const std::out_of_range&e){j["error"]=std::string("out_of_range:")+e.what();}catch(const std::logic_error&e){j["error"]=std::string("logic_error:")+e.what();}j["logs"]=logs;need(gb==values(gp)&&cb==values(cp),"input mutation");j["input_G"]=gb;j["input_C"]=cb;return j;}
int case_entry(int argc,char**argv){try{need(argc==4,"args");std::string id=argv[1];gsl_set_error_handler(handler);auto ctx=checked::Context::owned_isolated_process();arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");Json::Value out;out["id"]=id;
 if(id=="C01"){out["row"]=construct(gp,cp);need(out["row"]["error"]=="","valid constructor");}
 else if(id=="C02"){auto control=construct(gp,cp);for(unsigned n:{7,9}){arma::vec c(n);c.head(6)=cp;for(unsigned i=6;i<n;++i)c[i]=(i%2)?std::numeric_limits<double>::infinity():std::numeric_limits<double>::quiet_NaN();auto j=construct(gp,c);need(j["error"]==""&&j["model"]==control["model"]&&j["logs"]==control["logs"],"unused tail consumed");out["rows"].append(j);}}
 else if(id=="C03"){for(unsigned n=0;n<6;++n){arma::vec c=cp.head(n);auto j=construct(gp,c);need(j["error"]=="out_of_range:Mat::operator(): index out of bounds","short cell error");out["rows"].append(j);}}
 else if(id=="C04"){for(unsigned n:{0,13,15}){arma::vec g(n,arma::fill::ones);for(unsigned i=0;i<std::min<unsigned>(14,n);++i)g[i]=gp[i];auto j=construct(g,cp);need(j["error"]=="logic_error:Mat::init(): size is fixed and hence cannot be changed","indicator size error");out["rows"].append(j);}}
 else if(id=="C05"){arma::vec g=gp.head(13);for(unsigned n:{0,5}){arma::vec c=cp.head(n);auto j=construct(g,c);need(j["error"]=="out_of_range:Mat::operator(): index out of bounds","error ordering");out["rows"].append(j);}}
 else if(id=="C06"){arma::vec backing(20);backing.head(14)=gp;backing.tail(6)=cp;arma::vec g(backing.memptr(),14,false,true),c(backing.memptr()+14,6,false,true);out["row"]=construct(g,c);auto m=make(g,c);auto before=observe(*m);backing.fill(-99.);need(before==observe(*m),"shared backing retained");out["independent"]=before;}
 else if(id=="C07"){auto g=gp,c=cp;auto m=make(g,c);auto before=observe(*m);g.fill(-9.);c.fill(-8.);g.reset();c.reset();need(before==observe(*m),"owned snapshot failed");out["model"]=before;out["raw_reconstruction"]=construct(gp,cp);}
 else if(id=="C08"){auto m=make(gp,cp);checked::TestAccess::seed(*m);auto snap=checked::TestAccess::snapshot(*m);auto addr=checked::TestAccess::addresses(*m);checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};checked::testing::reset();checked::testing::fault().call=2;checked::testing::fault().status=checked::Status::backend_failed;auto fail=m->try_set_params_and_init(cell,ctx);need(!fail.report.committed&&snap==checked::TestAccess::snapshot(*m)&&addr==checked::TestAccess::addresses(*m),"rollback");checked::testing::reset();need(m->try_set_params_and_init(cell,ctx).report.committed,"checked update");out["after"]=observe(*m);for(auto x:m->Cparams)need(x==0.,"checked raw synchronization");}
 else need(false,"case");need(gsl_set_error_handler(handler)==handler,"handler restoration");out["ok"]=true;Json::StreamWriterBuilder writer;writer["indentation"]="";std::cout<<"RESULT "<<Json::writeString(writer,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"C01","C02","C03","C04","C05","C06","C07","C08"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
