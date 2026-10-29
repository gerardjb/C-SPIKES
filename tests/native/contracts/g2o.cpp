#include <armadillo>
#include <algorithm>
#include <sstream>
#include <memory>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <json/json.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <utility>
void need(bool x,const char* m){if(!x)throw std::runtime_error(m);}
void handler(const char*,const char*,int,int){}
std::string bits(double v){if(std::isnan(v))return "nan";if(std::isinf(v))return v>0?"+inf":"-inf";const char*h="0123456789abcdef";unsigned char b[8];std::memcpy(b,&v,8);std::string s;for(auto c:b){s+=h[c>>4];s+=h[c&15];}return s;}
template<class M>Json::Value values(const M& a){Json::Value j;j["rows"]=Json::UInt64(a.n_rows);j["cols"]=Json::UInt64(a.n_cols);j["values"]=Json::Value(Json::arrayValue);for(double v:a)j["values"].append(bits(v));return j;}
#ifdef G2O_BASELINE
using ExpectedVector=arma::vec;using ExpectedMatrix=arma::mat;
#else
using ExpectedVector=csp_model::HistoryVector;using ExpectedMatrix=csp_model::HistoryMatrix;
#endif
static_assert(std::is_same<decltype(std::declval<const GCaMP&>().getDFFValues()),ExpectedVector>::value,"DFF return");
static_assert(std::is_same<decltype(std::declval<const GCaMP&>().getBCaValues()),ExpectedVector>::value,"BCa return");
static_assert(std::is_same<decltype(std::declval<const GCaMP&>().getCaValues()),ExpectedVector>::value,"Ca return");
static_assert(std::is_same<decltype(std::declval<const GCaMP&>().getCaInValues()),ExpectedVector>::value,"CaIn return");
static_assert(std::is_same<decltype(std::declval<const GCaMP&>().getGValues()),ExpectedMatrix>::value,"G return");
Json::Value values(const csp_model::HistoryVector& a){Json::Value j;j["rows"]=Json::UInt64(a.size());j["cols"]=1;j["values"]=Json::Value(Json::arrayValue);for(std::size_t i=0;i<a.size();++i)j["values"].append(bits(a(i)));return j;}
Json::Value values(const csp_model::HistoryMatrix& a){Json::Value j;j["rows"]=Json::UInt64(a.rows());j["cols"]=Json::UInt64(a.cols());j["values"]=Json::Value(Json::arrayValue);for(std::size_t c=0;c<a.cols();++c)for(std::size_t r=0;r<a.rows();++r)j["values"].append(bits(a(r,c)));return j;}
std::size_t nrows(const arma::mat& m){return m.n_rows;}std::size_t ncols(const arma::mat& m){return m.n_cols;}
std::size_t nrows(const csp_model::HistoryMatrix& m){return m.rows();}std::size_t ncols(const csp_model::HistoryMatrix& m){return m.cols();}
template<class M>bool finite(const M& m){for(std::size_t c=0;c<ncols(m);++c)for(std::size_t r=0;r<nrows(m);++r)if(!std::isfinite(m(r,c)))return false;return true;}
Json::Value histories(const GCaMP& m){Json::Value j;j["fine"]=values(checked::TestAccess::fine(m));j["G"]=values(m.getGValues());j["BCa"]=values(m.getBCaValues());j["Ca"]=values(m.getCaValues());j["Ca_in"]=values(m.getCaInValues());return j;}
int case_entry(int argc,char**argv){try{
 need(argc==4,"arguments");std::string id=argv[1];gsl_set_error_handler(&handler);arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixtures");
 GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});m.setTimeStepMode(FIXEDLA);auto ctx=checked::Context::owned_isolated_process();checked::CellParameters cell{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]};Json::Value out;out["id"]=id;
 if(id=="O01"){out["history"]=histories(m);out["dff"]=values(m.getDFFValues());need(nrows(m.getGValues())*ncols(m.getGValues())==0&&checked::TestAccess::fine(m).n_elem==0,"empty matrices");GCaMP copy(m);need(histories(copy)==histories(m),"empty copy");}
 else if(id=="O02"){checked::TestAccess::seed(m);checked::TestAccess::pattern(m);auto saved=values(m.getDFFValues());auto other=histories(m);GCaMP copy(m);need(saved==values(copy.getDFFValues()),"deep copy values");checked::TestAccess::edit_copy(copy);need(saved==values(m.getDFFValues())&&saved!=values(copy.getDFFValues()),"copy independence");checked::TestAccess::self_assign(m);need(saved==values(m.getDFFValues())&&other==histories(m),"self assignment");auto returned=m.getDFFValues();returned(0)=999.;need(saved==values(m.getDFFValues()),"owned getter");auto returned_g=m.getGValues();returned_g(0,0)=-999.;auto returned_b=m.getBCaValues();returned_b(0)=-888.;auto returned_c=m.getCaValues();returned_c(0)=-777.;need(other==histories(m),"all getter mutation isolation");auto survived=[&](){GCaMP local(m);return local.getGValues();}();need(values(survived)==other["G"],"snapshot survives model destruction");out["dff"]=saved;out["copy"]=values(copy.getDFFValues());out["history"]=other;out["container"]=checked::TestAccess::container_contract();out["bridge"]=values(checked::TestAccess::defined_bridge(m));need(other==histories(m),"bridge non-DFF preservation");}
 else if(id=="O07"){
  checked::TestAccess::seed(m);const auto before=checked::TestAccess::snapshot(m);const auto refs=checked::TestAccess::addresses(m);const auto history=checked::TestAccess::histories(m);const auto hr=checked::TestAccess::history_addresses(m);
  checked::testing::reset();checked::testing::fault().call=2;checked::testing::fault().status=checked::Status::backend_failed;auto fail=m.try_set_params_and_init(cell,ctx);need(!fail.report.committed&&fail.report.status==checked::Status::backend_failed,"injected failure");need(before==checked::TestAccess::snapshot(m)&&refs==checked::TestAccess::addresses(m),"full rollback");checked::testing::reset();auto ok=m.try_set_params_and_init(cell,ctx);need(ok.report.committed&&ok.report.status==checked::Status::success,"checked success");need(history==checked::TestAccess::histories(m)&&hr==checked::TestAccess::history_addresses(m),"checked history preservation");out["history"]=histories(m);out["dff"]=values(m.getDFFValues());
 }else if(id=="O08"){throw std::runtime_error("retired method case is not admitted");
 }else{throw std::runtime_error("history-evolution cases covered by repaired endpoint suite");
 }
 need(gsl_set_error_handler(&handler)==&handler,"handler restoration");out["ok"]=true;Json::StreamWriterBuilder b;b["indentation"]="";std::cout<<"RESULT "<<Json::writeString(b,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"O01","O02","O07"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2],argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
