#include <armadillo>
#include <algorithm>
#include <sstream>
#include <memory>
#include "GCaMP_model.h"
#include <json/json.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#ifdef G2G_CANDIDATE
#include "state12_arma_bridge.h"
auto owned(const csp_model::State12& x){return csp_model::owned_arma(x);}
double* pointer(csp_model::State12& x){return x.data();}
#else
auto owned(const arma::vec::fixed<12>& x){return x;}
double* pointer(arma::vec::fixed<12>& x){return x.memptr();}
#endif
void need(bool x,const char* s){if(!x)throw std::runtime_error(s);}
std::string hex(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";unsigned char b[8];std::memcpy(b,&x,8);const char* h="0123456789abcdef";std::string s;for(auto c:b){s+=h[c>>4];s+=h[c&15];}return s;}
template<class M>Json::Value values(const M& m){Json::Value j(Json::arrayValue);for(auto x:m){need(std::isfinite(x),"nonfinite state control");j.append(hex(x));}return j;}
Json::Value record(GCaMP& m){Json::Value j;j["state"]=values(m.state);j["initial"]=values(m.initial_state);j["steps"]=m.step_count;j["dff"]=hex(m.getDFF());return j;}
int case_entry(int argc,char**argv){try{need(argc==4,"args");std::string id=argv[1];arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"params");GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});Json::Value out;out["id"]=id;
if(id=="S01"){out["records"].append(record(m));m.setState(5e-8);out["records"].append(record(m));m.setState(1e-7);out["records"].append(record(m));m.init();out["records"].append(record(m));}
else if(id=="S02") {auto before=values(m.state);auto copy=m.state;copy[0]=123.;need(values(m.state)==before,"owned copy alias");need(pointer(copy)!=pointer(m.state),"copy pointer");for(unsigned i=0;i<12;++i)need(&m.state[i]==pointer(m.state)+i,"contiguous layout");need(pointer(m.state)!=pointer(m.initial_state),"field alias");out["copy"]=values(copy);out["records"].append(record(m));out["owned"]=true;}
else if(id=="S03") {m.setTimeStepMode(FIXEDLA);arma::vec s=owned(m.state);m.evolve(0.,0,csp_model::ScalarInputView{s.memptr(),s.n_elem});out["valid"]=record(m);for(unsigned n:{11u,13u}) {auto before=record(m);auto address=pointer(m.state);arma::vec bad(n,arma::fill::zeros);std::string category="none",message;try{m.evolve(0.,0,csp_model::ScalarInputView{bad.memptr(),bad.n_elem});}catch(const std::logic_error& e){category="logic_error";message=e.what();}catch(const std::runtime_error& e){category="runtime_error";message=e.what();}need(category!="none","malformed accepted");need(record(m)==before&&address==pointer(m.state),"malformed changed state");Json::Value j;j["n"]=n;j["category"]=category;j["message"]=message;j["after"]=record(m);out["invalid"].append(j);}}
else if(id=="S04"){m.setTimeStepMode(FIXEDLA);auto before=record(m);double* address=pointer(m.state);arma::vec alias(address,12,false,true);m.evolve(0.,0,csp_model::ScalarInputView{alias.memptr(),alias.n_elem});need(before==record(m)&&address==pointer(m.state),"alias changed contract");out["records"].append(record(m));out["alias_values"]=values(alias);}
else if(id=="S05"){for(auto mode:{FIXED,FIXEDLA}){m.setState(1e-7);m.setTimeStepMode(mode);m.step_count=0;const double old=m.state[11];m.evolve(mode==FIXED?3e-6:1e-4,0);if(mode==FIXED)need(hex(m.state[11])==hex(old),"FIXED writeback changed");out["records"].append(record(m));}}
else if(id=="S06"){m.setState(1e-7);m.brightStates={8,6,5,3,2};out["reordered"]=hex(m.getDFF());m.brightStates={2,2,5,6,8};out["duplicate"]=hex(m.getDFF());out["records"].append(record(m));}
else need(false,"unknown");Json::StreamWriterBuilder b;b["indentation"]="";std::cout<<"RESULT "<<Json::writeString(b,out)<<"\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"S01","S02","S03","S04","S05","S06"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
