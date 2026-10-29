#include <armadillo>
#include <algorithm>
#include <sstream>
#include <memory>
#include "GCaMP_model.h"
#include <json/json.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#ifdef G2I_CANDIDATE
#include "parameter_block_arma_bridge.h"
template<std::size_t N> using Block=csp_model::ParameterBlock<N>;
template<std::size_t N> bool load(Block<N>& x,const std::string& p){return csp_model::load_raw_ascii(x,p);}
template<std::size_t N> void assign(Block<N>& x,const arma::vec& v){csp_model::assign_from_armadillo(x,v);}
#else
template<std::size_t N> using Block=typename arma::Col<double>::template fixed<N>;
template<class T> bool load(T& x,const std::string& p){return x.load(p,arma::raw_ascii);}
template<class T> void assign(T& x,const arma::vec& v){x=v;}
#endif
void need(bool x,const char* s){if(!x)throw std::runtime_error(s);}
std::string hex(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";unsigned char b[8];std::memcpy(b,&x,8);const char*h="0123456789abcdef";std::string s;for(auto c:b){s+=h[c>>4];s+=h[c&15];}return s;}
template<class T>Json::Value values(const T& x){Json::Value a(Json::arrayValue);for(auto v:x)a.append(hex(v));return a;}
template<class F> std::string category(F f){try{f();return "none";}catch(const std::out_of_range&){return "out_of_range";}catch(const std::logic_error&){return "logic_error";}catch(const std::runtime_error&){return "runtime_error";}}
template<std::size_t N>Json::Value blockcase(const std::string& id,const std::string& dir){Block<N>x;Json::Value out;
if(id=="Q01"){for(auto v:x)need(v==0.,"default not zero");out["default"]=values(x);auto copy=x;copy[0]=123.;need(x[0]==0.&&&x[0]!=&copy[0],"copy alias");for(size_t i=0;i<N;++i)need(&x[i]==&x[0]+i,"layout");out["copy"]=values(copy);out["bounds"]=category([&]{(void)x(N);});need(out["bounds"].asString()=="out_of_range","bounds mismatch");return out;}
if(id=="Q11"){arma::vec v(N);for(size_t i=0;i<N;++i)v[i]=i+.25;assign(x,v);out["valid"]=values(x);for(unsigned len:{unsigned(N-1),unsigned(N+1)}){auto before=values(x);arma::vec bad(len,arma::fill::zeros);auto c=category([&]{assign(x,bad);});need(c=="logic_error"&&values(x)==before,"shape assignment changed");Json::Value r;r["length"]=len;r["category"]=c;r["after"]=values(x);out["invalid"].append(r);}return out;}
std::vector<std::string> labels;
if(id=="Q02")labels={"valid"};else if(id=="Q03")labels={"missing"};else if(id=="Q04")labels={"empty"};else if(id=="Q05")labels={"short","long"};else if(id=="Q06")labels={"row"};else if(id=="Q07")labels={"inconsistent"};else if(id=="Q08")labels={"bad_token"};else if(id=="Q09")labels={"nonfinite"};else if(id=="Q10")labels={"prefix_token","blank_tail"};else need(false,"unknown block case");
for(auto label:labels){x.fill(77.);bool ok=load(x,dir+"/"+std::to_string(N)+"_"+label+".txt");bool expected=label=="valid"||label=="nonfinite"||label=="prefix_token"||label=="blank_tail";need(ok==expected,"loader return differs from frozen source prediction");if(!ok)for(auto v:x)need(v==0.,"failure not zero");Json::Value r;r["label"]=label;r["ok"]=ok;r["values"]=values(x);out.append(r);}return out;}
Json::Value model_record(GCaMP&m){Json::Value o;o["Gparams"]=values(m.Gparams);o["Cparams"]=values(m.Cparams);o["state"]=values(m.state);o["initial"]=values(m.initial_state);o["DFF"]=hex(m.getDFF());o["mode"]=int(m.TSMode);GCaMP_params p;m.read_params(p);o["handoff_gp"]=values(p.Gparams);for(double x:{p.G_tot,p.gamma,p.DCaT,p.Rf,p.gam_in,p.gam_out})o["scalars"].append(hex(x));auto before=values(m.Gparams);p.Gparams[0]=123.;need(values(m.Gparams)==before,"snapshot alias");return o;}
int case_entry(int argc,char**argv){try{need(argc==5,"args");std::string id=argv[1];Json::Value o;o["id"]=id;
if(id!="Q12"){o["14"]=blockcase<14>(id,argv[4]);o["6"]=blockcase<6>(id,argv[4]);}
if(id=="Q11"||id=="Q12"){arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixture params");
if(id=="Q11"){arma::vec shortcp(5,arma::fill::zeros);o["short_c"]=category([&]{GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{shortcp.memptr(),shortcp.n_elem});});need(o["short_c"].asString()=="out_of_range","short C category");for(unsigned n:{13u,15u}){arma::vec bad(n,arma::fill::zeros);auto c=category([&]{GCaMP m(csp_model::ScalarInputView{bad.memptr(),bad.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});});need(c=="logic_error","wrong G category");o["wrong_g"].append(c);}}
else {GCaMP a{std::string{},std::string{}};GCaMP b(cp[0],cp[1],cp[2],cp[3],cp[4],cp[5],argv[2]);GCaMP c(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});GCaMP d{std::string(argv[2]),std::string(argv[3])};arma::vec tail(8);for(size_t i=0;i<6;++i)tail[i]=cp[i];tail[6]=123.;tail[7]=456.;GCaMP e(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{tail.memptr(),tail.n_elem});
for(auto*m:{&a,&b,&c,&d,&e}){o["before"].append(model_record(*m));auto raw=values(m->Cparams);m->setParams(cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]);need(values(m->Cparams)==raw,"raw record synchronized");o["after"].append(model_record(*m));}
need(o["before"][2]==o["before"][4],"C tail behavior");for(auto*m:{&a,&b,&c,&e})for(auto v:m->Cparams)need(v==0.,"unassigned raw record not zero");need(values(d.Cparams)==values(cp),"file raw record");}}
Json::StreamWriterBuilder b;b["indentation"]="";std::cout<<"RESULT "<<Json::writeString(b,o)<<"\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"Q01","Q02","Q03","Q04","Q05","Q06","Q07","Q08","Q09","Q10","Q11","Q12"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(5,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
