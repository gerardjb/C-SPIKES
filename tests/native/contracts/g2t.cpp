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
#ifdef G2T_BASELINE
const arma::vec& input(const arma::vec& x){return x;}
arma::vec& output(arma::vec& x){return x;}
#else
csp_model::ScalarInputView input(const arma::vec& x){return {x.memptr(),x.n_elem};}
csp_model::ScalarOutputView output(arma::vec& x){return {x.memptr(),x.n_elem};}
static_assert(std::is_same<decltype(static_cast<void(GCaMP::*)(double,int,csp_model::ScalarInputView)>(&GCaMP::evolve)),void(GCaMP::*)(double,int,csp_model::ScalarInputView)>::value,"supplied evolve signature");
static_assert(std::is_same<decltype(&GCaMP::fixedStep_LA_threadsafe),double(GCaMP::*)(double,int,csp_model::ScalarInputView,csp_model::ScalarOutputView)>::value,"LA signature");
static_assert(std::is_same<decltype(&GCaMP::evolve_threadsafe),void(GCaMP::*)(double,int,csp_model::ScalarInputView,csp_model::ScalarOutputView,double&)>::value,"wrapper signature");
#endif
#ifndef ARMA_CHECK_CONFORMANCE
#error baseline conformance must be enabled
#endif
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
std::string bits(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";const char*h="0123456789abcdef";unsigned char a[8];std::memcpy(a,&x,8);std::string s;for(auto c:a){s+=h[c>>4];s+=h[c&15];}return s;}
template<class V>Json::Value values(const V& v){Json::Value j(Json::arrayValue);for(auto x:v)j.append(bits(x));return j;}
Json::Value logs(Json::arrayValue);
void capture(const csp_diagnostics::Record&r){Json::Value j;j["operation"]=r.operation;j["method"]=r.first.method;j["field"]=r.first.field;j["mode"]=r.first.mode;j["index"]=Json::Int64(r.first.index);j["value"]=bits(r.first.value);j["count"]=Json::UInt64(r.detections);j["complete"]=r.complete;for(auto c:r.counts)j["counts"].append(Json::UInt64(c));logs.append(j);}
void handler(const char*,const char*,int,int){}
Json::Value model(const GCaMP&m){Json::Value j;j["state"]=values(m.state);j["initial"]=values(m.initial_state);j["DFF"]=bits(m.DFF);j["step"]=m.step_count;j["mode"]=int(m.TSMode);j["rates"]=values(checked::TestAccess::rates(m));j["gparams"]=values(m.Gparams);j["cparams"]=values(m.Cparams);Json::Value mat(Json::arrayValue);for(unsigned i=0;i<81;++i)mat.append(bits(m.Gmat.data()[i]));j["matrix"]=mat;return j;}
template<class F>Json::Value call(GCaMP&m,F f){Json::Value j;j["before"]=model(m);logs=Json::Value(Json::arrayValue);try{csp_diagnostics::Scope scope("evolution-probe",capture);f();j["error"]="";}catch(const std::out_of_range&e){j["error"]=std::string("out_of_range:")+e.what();}catch(const std::logic_error&e){j["error"]=std::string("logic_error:")+e.what();}j["after"]=model(m);j["logs"]=logs;return j;}
arma::vec pattern(unsigned n){arma::vec v(n);for(unsigned i=0;i<n;++i)v[i]=(i+1)*1e-8;if(n>9)v[9]=1e-5;if(n>10)v[10]=1e-7;if(n>11)v[11]=2e-8;return v;}
arma::vec sentinel(unsigned n){arma::vec v(n);for(unsigned i=0;i<n;++i)v[i]=-20.-i;return v;}
Json::Value run(GCaMP&m,arma::vec& in,arma::vec& out,double dt,int ns,bool wrapper,double& dff){auto ip=values(in),op=values(out);auto ptr=out.memptr();auto n=out.n_elem;Json::Value j=call(m,[&]{if(wrapper)m.evolve_threadsafe(dt,ns,input(in),output(out),dff);else dff=m.fixedStep_LA_threadsafe(dt,ns,input(in),output(out));});need(ptr==out.memptr()&&n==out.n_elem,"output resized");j["input_before"]=ip;j["input_after"]=values(in);j["output_before"]=op;j["output_after"]=values(out);j["dff"]=bits(dff);return j;}
int case_entry(int argc,char**argv){try{need(argc==4,"args");std::string id=argv[1];gsl_set_error_handler(handler);auto ctx=checked::Context::owned_isolated_process();arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});m.setTimeStepMode(FIXEDLA);m.DFF=0.;Json::Value out;out["id"]=id;
 if(id=="T01"||id=="T02"){for(bool wrapper:{false,true})for(int ns:{0,1}){auto a=pattern(12),b=sentinel(12);double d=-99.;auto j=run(m,a,b,id=="T01"?0.:.0001,ns,wrapper,d);need(j["error"]=="","valid LA");need(j["input_before"]==j["input_after"],"input mutated");out["rows"].append(j);}}
 else if(id=="T03"){for(auto mode:{FIXEDLA,FIXED})for(double dt:{0.,.000003,.0003}){m.setTimeStepMode(mode);auto a=pattern(12);auto p=m.state.data();auto j=call(m,[&]{m.evolve(dt,1,input(a));});need(j["error"]==""&&p==m.state.data(),"supplied dispatch");out["rows"].append(j);}}
 else if(id=="T04"){for(unsigned n:{0,11,13}){auto a=pattern(n);if(n)a[n-1]=std::numeric_limits<double>::quiet_NaN();auto snap=checked::TestAccess::snapshot(m);auto addr=checked::TestAccess::addresses(m);auto j=call(m,[&]{m.evolve(.0001,1,input(a));});need(j["error"]=="logic_error:Mat::init(): size is fixed and hence cannot be changed","length error");need(snap==checked::TestAccess::snapshot(m)&&addr==checked::TestAccess::addresses(m),"bad length mutated");out["rows"].append(j);}}
 else if(id=="T05"){for(double dt:{0.,.0001}){arma::vec a(m.state.data(),12,false,true);auto ptr=m.state.data();auto j=call(m,[&]{m.evolve(dt,1,input(a));});need(j["error"]==""&&ptr==m.state.data(),"supplied alias");out["rows"].append(j);}}
 else if(id=="T06"){for(unsigned n=0;n<12;++n){auto a=pattern(n),b=sentinel(12);auto snap=checked::TestAccess::snapshot(m);double d=-99.;auto j=run(m,a,b,.0001,1,false,d);need(j["error"]==(n<9?"out_of_range:Col::subvec(): indices out of bounds or incorrectly used":"out_of_range:Mat::operator(): index out of bounds"),"input error");need(j["output_before"]==j["output_after"]&&snap==checked::TestAccess::snapshot(m)&&d==-99.,"short input mutation");out["rows"].append(j);}}
 else if(id=="T07"){auto a=pattern(15),b=sentinel(12);a[13]=std::numeric_limits<double>::quiet_NaN();a[14]=std::numeric_limits<double>::infinity();double d=-99.;out["row"]=run(m,a,b,.0001,1,true,d);auto c=pattern(12),e=sentinel(12);double f=-99.;out["control"]=run(m,c,e,.0001,1,true,f);need(values(b)==values(e)&&bits(d)==bits(f),"tail used numerically");}
 else if(id=="T08"){auto a=pattern(12),full=sentinel(12);double ref=-99.;run(m,a,full,.0001,1,true,ref);for(unsigned n=0;n<12;++n){auto b=sentinel(n);double d=-99.;auto j=run(m,a,b,.0001,1,true,d);need(j["error"]=="out_of_range:Mat::operator(): index out of bounds"&&d==-99.,"short output error/publication");for(unsigned i=0;i<n;++i)need(bits(b[i])==bits(full[i]),"partial output prefix");out["rows"].append(j);}}
 else if(id=="T09"){auto a=pattern(12),b=sentinel(15);double d=-99.;out["row"]=run(m,a,b,.0001,1,true,d);for(unsigned i=12;i<15;++i)need(b[i]==-20.-i,"tail changed");}
 else if(id=="T10"){for(bool wrap:{false,true}){auto a=pattern(12);double d=-99.;out["rows"].append(run(m,a,a,.0001,1,wrap,d));}}
 else if(id=="T11"){for(unsigned shift:{1,4}){auto backing=pattern(20);arma::vec a(backing.memptr(),12,false,true),b(backing.memptr()+shift,12,false,true);auto initial=values(backing);double d=-99.;auto j=run(m,a,b,.0001,1,true,d);j["backing_before"]=initial;j["backing_after"]=values(backing);out["rows"].append(j);}}
 else if(id=="T12"){for(bool same:{false,true}){auto a=pattern(12);arma::vec b(m.state.data(),12,false,true);double d=-99.;out["rows"].append(run(m,same?b:a,b,.0001,1,true,d));}}
 else if(id=="T13"){auto a=pattern(12),b=sentinel(12);out["output_alias"]=run(m,a,b,.0001,1,true,b[0]);need(out["output_alias"]["dff"]==out["output_alias"]["output_after"][0],"dff output alias ordering");out["model_alias"]=run(m,a,b,.0001,1,true,m.DFF);}
 else if(id=="T14"){m.setTimeStepMode(FIXED);auto a=pattern(12),b=sentinel(12);double d=-99.;auto j=run(m,a,b,.000003,1,true,d);need(j["output_before"]==j["output_after"]&&d==-99.,"FIXED output mutation");need(j["before"]["step"]!=j["after"]["step"],"FIXED not dispatched");out["row"]=j;}
 else if(id=="T15"){for(double v:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){auto a=pattern(12),b=sentinel(12);a[2]=v;double d=-99.;out["nonfinite"].append(run(m,a,b,0.,0,true,d));}m.brightStates={8,2,2,5,3};auto a=pattern(12),b=sentinel(12);double d=-99.;out["brightness"]=run(m,a,b,.0001,0,true,d);checked::TestAccess::invalidate_normalization(m);out["normalization"]=run(m,a,b,0.,0,true,d);}
 else if(id=="T16"){auto a=pattern(12),b=sentinel(12);double d=-99.;out["before"]=run(m,a,b,.0001,1,true,d);auto snap=checked::TestAccess::snapshot(m);auto addr=checked::TestAccess::addresses(m);checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};checked::testing::reset();checked::testing::fault().call=1;checked::testing::fault().status=checked::Status::backend_failed;auto fail=m.try_set_params_and_init(cell,ctx);need(!fail.report.committed&&snap==checked::TestAccess::snapshot(m)&&addr==checked::TestAccess::addresses(m),"rollback");checked::testing::reset();need(m.try_set_params_and_init(cell,ctx).report.committed,"checked update");out["after"]=run(m,a,b,.0001,1,true,d);}
 else need(false,"case");need(gsl_set_error_handler(handler)==handler,"handler restoration");out["ok"]=true;Json::StreamWriterBuilder writer;writer["indentation"]="";std::cout<<"RESULT "<<Json::writeString(writer,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"T01","T02","T03","T04","T05","T06","T07","T08","T09","T10","T11","T12","T13","T14","T15","T16"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
