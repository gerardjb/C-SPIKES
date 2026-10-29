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
#ifdef G2V_BASELINE
const arma::vec& input(const arma::vec& x){return x;}
#else
csp_model::ScalarInputView input(const arma::vec& x){return {x.memptr(),x.n_elem};}
static_assert(std::is_same<decltype(&GCaMP::integrateOverTime),void(GCaMP::*)(csp_model::ScalarInputView,csp_model::ScalarInputView)>::value,"primary signature");
// Alternative signature absence is asserted in this translation unit.
#endif
static_assert(sizeof(arma::uword)==sizeof(std::size_t)&&sizeof(arma::uword)==8,"qualified widths");
#ifndef ARMA_CHECK_CONFORMANCE
#error baseline conformance must be enabled
#endif
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
std::string bits(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";const char*h="0123456789abcdef";unsigned char a[8];std::memcpy(a,&x,8);std::string s;for(auto c:a){s+=h[c>>4];s+=h[c&15];}return s;}
template<class V>Json::Value values(const V& v){Json::Value j(Json::arrayValue);for(auto x:v)j.append(bits(x));return j;}
Json::Value vec(const csp_model::HistoryVector&v){Json::Value j(Json::arrayValue);for(std::size_t i=0;i<v.size();++i)j.append(bits(v(i)));return j;}
Json::Value mat(const csp_model::HistoryMatrix&v){Json::Value j;j["rows"]=Json::UInt64(v.rows());j["cols"]=Json::UInt64(v.cols());j["values"]=Json::Value(Json::arrayValue);for(std::size_t c=0;c<v.cols();++c)for(std::size_t r=0;r<v.rows();++r)j["values"].append(bits(v(r,c)));return j;}
Json::Value history(const GCaMP&m){Json::Value j;j["G"]=mat(m.getGValues());j["BCa"]=vec(m.getBCaValues());j["Ca"]=vec(m.getCaValues());j["Ca_in"]=vec(m.getCaInValues());auto fine=checked::TestAccess::fine(m);j["fine_rows"]=Json::UInt64(fine.n_rows);j["fine_cols"]=Json::UInt64(fine.n_cols);j["fine_values"]=values(fine);j["dff_size"]=Json::UInt64(checked::TestAccess::dff_size(m));return j;}
Json::Value model(const GCaMP&m){Json::Value j;j["state"]=values(m.state);j["initial"]=values(m.initial_state);j["DFF"]=bits(m.DFF);j["step"]=m.step_count;j["mode"]=int(m.TSMode);j["rates"]=values(checked::TestAccess::rates(m));j["gparams"]=values(m.Gparams);j["cparams"]=values(m.Cparams);Json::Value a(Json::arrayValue);for(unsigned i=0;i<81;++i)a.append(bits(m.Gmat.data()[i]));j["matrix"]=a;return j;}
Json::Value logs(Json::arrayValue);
void capture(const csp_diagnostics::Record&r){Json::Value j;j["operation"]=r.operation;j["method"]=r.first.method;j["field"]=r.first.field;j["mode"]=r.first.mode;j["index"]=Json::Int64(r.first.index);j["value"]=bits(r.first.value);j["count"]=Json::UInt64(r.detections);j["complete"]=r.complete;for(auto c:r.counts)j["counts"].append(Json::UInt64(c));logs.append(j);}
void handler(const char*,const char*,int,int){}
Json::Value run(GCaMP&m,arma::vec&t,arma::vec&s,bool alt=false){Json::Value j;j["before"]=model(m);j["histories_before"]=history(m);j["time_before"]=values(t);j["spikes_before"]=values(s);auto addr=checked::TestAccess::addresses(m);auto state=m.state;auto initial=m.initial_state;auto count=m.step_count;auto dff=m.DFF;logs=Json::Value(Json::arrayValue);
 try{csp_diagnostics::Scope scope("history-input-probe",capture);need(!alt,"retired alternative not admitted");m.integrateOverTime(input(t),input(s));j["error"]="";}
 catch(const std::out_of_range&e){j["error"]=std::string("out_of_range:")+e.what();}
 catch(const std::logic_error&e){j["error"]=std::string("logic_error:")+e.what();}
 catch(const std::runtime_error&e){j["error"]=std::string("runtime_error:")+e.what();}
 j["logs"]=logs;j["after"]=model(m);j["histories_after"]=history(m);j["time_after"]=values(t);j["spikes_after"]=values(s);auto after=checked::TestAccess::addresses(m);for(unsigned i=0;i<6;++i)need(addr[i]==after[i],"fixed storage moved");need(values(state)==values(m.state)&&values(initial)==values(m.initial_state)&&count==m.step_count&&bits(dff)==bits(m.DFF),"model writeback changed");return j;}
int case_entry(int argc,char**argv){try{need(argc==4,"args");std::string id=argv[1];gsl_set_error_handler(handler);auto ctx=checked::Context::owned_isolated_process();arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});m.setTimeStepMode(FIXEDLA);m.DFF=0.;checked::TestAccess::seed(m);arma::vec t={0.,.0001,.0002,.001},s;Json::Value out;out["id"]=id;
 if(id=="V01"){arma::vec empty;auto old=history(m);auto j=run(m,empty,s);need(j["error"]=="out_of_range:Mat::operator(): index out of bounds","empty time error");need(j["histories_after"]["dff_size"].asUInt64()==0,"DFF not reset");need(j["histories_after"]["G"]==old["G"]&&j["histories_after"]["fine_rows"].asUInt64()==0&&j["histories_after"]["BCa"].size()==0,"primary reset order");out["row"]=j;}
 else if(id=="V03"){out["row"]=run(m,t,s);}
 else if(id=="V04"){s={.0005,.0001,.0005};out["row"]=run(m,t,s);}
 else if(id=="V06"){for(bool alt:{false})for(unsigned n:{1,3}){arma::vec eq(n);eq.fill(.0001);out["rows"].append(run(m,eq,s,alt));}}
 else if(id=="V07"){t={0.,.0002,-.0001,.002,.0001,.0001,.001};out["rows"].append(run(m,t,s));}
 else if(id=="V08"){t={0.,std::numeric_limits<double>::quiet_NaN(),.0002,.001};out["rows"].append(run(m,t,s));}
 else if(id=="V09"){for(bool alt:{false}){out["rows"].append(run(m,t,s,alt));s={std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),.0005};out["rows"].append(run(m,t,s,alt));s.reset();}}
 else if(id=="V10"){for(bool alt:{false})out["rows"].append(run(m,t,t,alt));}
 else if(id=="V11"){arma::vec backing={0.,.0001,.0002,.001,.0005};arma::vec times(backing.memptr(),4,false,true),spikes(backing.memptr()+1,3,false,true);for(bool alt:{false}){auto b=values(backing);out["rows"].append(run(m,times,spikes,alt));need(values(backing)==b,"external input mutation");}}
 else if(id=="V12"){for(bool alt:{false}){for(unsigned i=0;i<4;++i)m.Gmat.data()[i]=t[i];arma::vec times(m.Gmat.data(),4,false,true);auto j=run(m,times,s,alt);need(j["time_before"]!=j["time_after"],"live generator time alias not exercised");out["rows"].append(j);}}
 else if(id=="V13"){for(bool alt:{false}){m.Gmat.data()[0]=.0005;m.Gmat.data()[1]=.0001;arma::vec spikes(m.Gmat.data(),2,false,true);out["rows"].append(run(m,t,spikes,alt));}}
 else if(id=="V14"){for(bool initial:{false,true}){double*p=initial?m.initial_state.data():m.state.data();for(unsigned i=0;i<4;++i)p[i]=t[i];arma::vec times(p,4,false,true);out["rows"].append(run(m,times,s));}}
 else if(id=="V15"){auto snap=checked::TestAccess::snapshot(m);auto addr=checked::TestAccess::addresses(m);checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};checked::testing::reset();checked::testing::fault().call=2;checked::testing::fault().status=checked::Status::backend_failed;auto fail=m.try_set_params_and_init(cell,ctx);need(!fail.report.committed&&snap==checked::TestAccess::snapshot(m)&&addr==checked::TestAccess::addresses(m),"rollback");checked::testing::reset();auto h=checked::TestAccess::histories(m);auto refs=checked::TestAccess::history_addresses(m);need(m.try_set_params_and_init(cell,ctx).report.committed&&h==checked::TestAccess::histories(m)&&refs==checked::TestAccess::history_addresses(m),"checked history preservation");out["row"]=run(m,t,s);}
 else if(id=="V16"){out["first"]=run(m,t,s);auto saved=m.getGValues();auto exact=mat(saved);arma::vec shorter=t.head(3);out["repeat"]=run(m,shorter,s);need(mat(saved)==exact,"owned getter changed");out["defined_bridge"]=vec(checked::TestAccess::defined_bridge(m));}
 else need(false,"case");need(gsl_set_error_handler(handler)==handler,"handler restoration");out["ok"]=true;Json::StreamWriterBuilder writer;writer["indentation"]="";std::cout<<"RESULT "<<Json::writeString(writer,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"V01","V03","V04","V06","V07","V08","V09","V10","V11","V12","V13","V14","V15","V16"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}

// A removed history implementation must not silently reappear as an alias.
template<class T, class = void> struct csp_has_alternative_history : std::false_type {};
template<class T> struct csp_has_alternative_history<T, std::void_t<decltype(&T::integrateOverTime2)>> : std::true_type {};
static_assert(!csp_has_alternative_history<GCaMP>::value, "alternative history API is retired");
