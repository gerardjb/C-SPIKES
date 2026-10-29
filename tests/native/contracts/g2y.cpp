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
#include <iostream>
#include <cstring>
#include <cmath>
#include <memory>
#ifdef G2Y_BASELINE
static_assert(std::is_same<decltype(std::declval<GCaMP&>().steady_state(0.)),arma::vec>::value,"baseline return");
const double* data(const arma::vec&v){return v.memptr();}
#else
static_assert(std::is_same<decltype(std::declval<GCaMP&>().steady_state(0.)),csp_model::State12>::value,"owned return");
static_assert(std::is_copy_constructible<csp_model::State12>::value,"owned copy");
const double* data(const csp_model::State12&v){return v.data();}
#endif
#ifndef ARMA_CHECK_CONFORMANCE
#error conformance required
#endif
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
std::string bits(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";const char*h="0123456789abcdef";unsigned char a[8];std::memcpy(a,&x,8);std::string s;for(auto c:a){s+=h[c>>4];s+=h[c&15];}return s;}
std::string hex(const std::string&s){const char*h="0123456789abcdef";std::string out;for(unsigned char c:s){out+=h[c>>4];out+=h[c&15];}return out;}
template<class V>Json::Value values(const V&v){Json::Value j(Json::arrayValue);for(auto x:v)j.append(bits(x));return j;}
Json::Value logs(Json::arrayValue);
void capture(const csp_diagnostics::Record&r){Json::Value j;j["operation"]=r.operation;j["method"]=r.first.method;j["field"]=r.first.field;j["mode"]=r.first.mode;j["index"]=Json::Int64(r.first.index);j["value"]=bits(r.first.value);j["count"]=Json::UInt64(r.detections);j["complete"]=r.complete;for(auto c:r.counts)j["counts"].append(Json::UInt64(c));logs.append(j);}
void handler(const char*,const char*,int,int){}
Json::Value observe(const GCaMP&m){Json::Value j;j["state"]=values(m.state);j["initial"]=values(m.initial_state);j["rates"]=values(checked::TestAccess::rates(m));j["generator"]=values(m.Gmat);j["norms"]=values(checked::TestAccess::norms(m));j["scalars"]=values(checked::TestAccess::cell_scalars(m));j["snapshot"]=hex(checked::TestAccess::snapshot(m));return j;}
std::unique_ptr<GCaMP> make(const arma::vec&gp,const arma::vec&cp){auto m=std::make_unique<GCaMP>(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});m->DFF=.125;m->step_count=7;checked::TestAccess::seed(*m);return m;}
int case_entry(int argc,char**argv){try{need(argc==4,"args");std::string id=argv[1];gsl_set_error_handler(handler);auto ctx=checked::Context::owned_isolated_process();arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");auto m=make(gp,cp);Json::Value out;out["id"]=id;
 if(id=="E01"||id=="E02"||id=="E03"||id=="E04"){
  const double ca[]={0.,5e-8,1e-6,1e-2};unsigned i=std::stoul(id.substr(1))-1;if(i==3)m->TSMode=FIXEDLA;
  auto unchanged=checked::TestAccess::without_cache(*m);auto addresses=checked::TestAccess::addresses(*m);
  {csp_diagnostics::Scope outer("equilibrium-probe",capture);outer.mode(FIXEDLA);auto x=m->steady_state(ca[i]);need(csp_diagnostics::active()->record.current.mode==FIXEDLA,"context restoration");need(x.n_elem==12,"size");need(data(x)!=m->state.data()&&data(x)!=m->initial_state.data()&&data(x)!=m->Gmat.data(),"return aliases model");need(bits(x[9])==bits(checked::TestAccess::buffer_initial(*m))&&bits(x[10])==bits(ca[i])&&bits(x[11])==bits(ca[i]),"tail mapping");out["result"]=values(x);}
  need(unchanged==checked::TestAccess::without_cache(*m)&&addresses==checked::TestAccess::addresses(*m),"unexpected direct publication");out["model"]=observe(*m);
 }
 else if(id=="E05"){
  auto x=m->steady_state(1e-6);auto retained=x;auto original=values(x);auto before=checked::TestAccess::snapshot(*m);x[0]=-777.;need(before==checked::TestAccess::snapshot(*m)&&values(retained)==original&&data(x)!=data(retained),"copy alias");auto y=m->steady_state(1e-6);need(values(y)==original,"repeat");m->setState(5e-8);need(values(retained)==original,"model mutation aliases return");m.reset();need(values(retained)==original,"lifetime");out["retained"]=values(retained);out["edited"]=values(x);out["repeat"]=values(y);
 }
 else if(id=="E06"){
  out["initialization"]=observe(*m);m->brightStates={8,2,2,6,3};m->setParams(cp[0]*1.25,cp[1]+1.,cp[2],cp[3],cp[4],cp[5]);out["parameters"]=observe(*m);auto x=m->steady_state(1e-6);auto norms=checked::TestAccess::norms(*m);m->setState(1e-6);need(values(m->state)==values(x)&&values(m->initial_state)==values(x)&&values(norms)==values(checked::TestAccess::norms(*m)),"setState mapping");out["state"]=observe(*m);
 }
 else if(id=="E07"){
  m->steady_state(1e-6);m->brightStates[4]=12;auto state=values(m->state),initial=values(m->initial_state),norms=values(checked::TestAccess::norms(*m));auto histories=checked::TestAccess::histories(*m);auto addr=checked::TestAccess::addresses(*m);std::string error;
  {csp_diagnostics::Scope outer("failed-parameters",capture);try{m->setParams(cp[0]*1.25,cp[1]+1.,cp[2],cp[3],cp[4],cp[5]);}catch(const std::out_of_range&e){error=std::string("out_of_range:")+e.what();}}
  need(!error.empty(),"missing bounds error");need(state==values(m->state)&&initial==values(m->initial_state)&&norms==values(checked::TestAccess::norms(*m))&&histories==checked::TestAccess::histories(*m)&&addr==checked::TestAccess::addresses(*m),"failure publication");need(bits(checked::TestAccess::cell_scalars(*m)[0])==bits(cp[0]*1.25)&&bits(checked::TestAccess::cell_scalars(*m)[1])==bits(cp[1]+1.),"scalars did not publish");out["error"]=error;out["after"]=observe(*m);
 }
 else if(id=="E08"){
  auto x=m->steady_state(1e-6);auto xbefore=values(x);auto before=checked::TestAccess::snapshot(*m);auto addr=checked::TestAccess::addresses(*m);checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};checked::testing::reset();checked::testing::fault().call=2;checked::testing::fault().status=checked::Status::backend_failed;auto fail=m->try_set_params_and_init(cell,ctx);need(!fail.report.committed&&before==checked::TestAccess::snapshot(*m)&&addr==checked::TestAccess::addresses(*m),"rollback");checked::testing::reset();need(m->try_set_params_and_init(cell,ctx).report.committed,"checked success");need(values(x)==xbefore,"return not independent");out["saved"]=values(x);out["after"]=observe(*m);{csp_diagnostics::Scope outer("post-checked",capture);outer.mode(FIXEDLA);auto y=m->steady_state(0.);out["result"]=values(y);need(csp_diagnostics::active()->record.current.mode==FIXEDLA,"restoration after checked");}
 }
 else need(false,"case");need(gsl_set_error_handler(handler)==handler,"handler restoration");out["logs"]=logs;out["ok"]=true;Json::StreamWriterBuilder writer;writer["indentation"]="";std::cout<<"RESULT "<<Json::writeString(writer,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"E01","E02","E03","E04","E05","E06","E07","E08"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
