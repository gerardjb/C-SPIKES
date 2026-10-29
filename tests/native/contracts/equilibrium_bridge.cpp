#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include "GCaMP_model.h"
#include "checked_equilibrium.h"
#include "test_access.h"
#include <armadillo>
#ifndef G2AB_BASELINE
#include "equilibrium_numpy_bridge.h"
#endif
#include <gsl/gsl_errno.h>
#include <json/json.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <limits>
namespace py=pybind11;
void need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
void handler(const char*,const char*,int,int){}
void restored(){need(gsl_set_error_handler(&handler)==&handler,"handler restoration");}
py::object publish(const std::array<double,12>& a){
#ifdef G2AB_BASELINE
 arma::vec v(12);std::copy(a.begin(),a.end(),v.begin());return py::cast(v);
#else
 return py::cast(csp_python::EquilibriumOutput12{a});
#endif
}
std::string bytes(const std::array<double,12>& a){return std::string(reinterpret_cast<const char*>(a.data()),sizeof(a));}
std::string hex(const std::string& a){const char*h="0123456789abcdef";std::string r;for(unsigned char c:a){r+=h[c>>4];r+=h[c&15];}return r;}
Json::Value inspect(py::object obj,const std::array<double,12>& expected){auto a=obj.cast<py::array_t<double>>();need(a.ndim()==1&&a.shape(0)==12&&a.strides(0)==8&&a.owndata()&&a.writeable()&&a.attr("base").is_none(),"array contract");need(std::memcmp(a.data(),expected.data(),96)==0,"exact copy");return hex(bytes(expected));}
GCaMP* nested_model=nullptr;bool nested_ok=false;
void nested(){const auto before=checked::TestAccess::snapshot(*nested_model);auto r=nested_model->try_steady_state(1e-6,checked::Context::owned_isolated_process());py::object state=py::none();if(r.state)state=publish(*r.state);nested_ok=r.report.status==checked::Status::context_busy&&!r.report.committed&&r.report.solve_count==0&&!r.state&&state.is_none()&&before==checked::TestAccess::snapshot(*nested_model);}
int main(int argc,char**argv){try{
 need(argc==4,"args");setenv("PYTHONHOME",G2AB_PYTHON_HOME,1);py::scoped_interpreter interpreter{};std::string id=argv[1];Json::Value out;out["id"]=id;
 if(id=="N01"){
  std::array<double,12>a{0.,-0.,1.,-1.,1e-300,1e300,std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),7.,8.,9.};const auto expected=a;auto x=publish(a);auto y=publish(a);out["first"]=inspect(x,expected);a.fill(-77.);out["independent"]=inspect(y,expected);x.cast<py::array_t<double>>().mutable_at(0)=-999.;out["separate"]=inspect(y,expected);py::object survived;{auto transient=expected;survived=publish(transient);}out["lifetime"]=inspect(survived,expected);
 }else if(id=="N02"){
  gsl_set_error_handler(&handler);arma::vec gp,cp;need(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii)&&gp.n_elem==14&&cp.n_elem==6,"fixtures");std::array<double,14> raw;std::copy(gp.begin(),gp.end(),raw.begin());auto ctx=checked::Context::owned_isolated_process();auto made=checked::try_create_gcamp(raw,{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]},ctx);need(bool(made.model),"factory");nested_model=made.model.get();checked::testing::reset();checked::testing::fault().nested=&nested;auto r=nested_model->try_steady_state(1e-6,ctx);checked::testing::reset();need(nested_ok&&r.report.committed&&r.state.has_value(),"busy context/outer success");restored();out["outer"]=inspect(publish(*r.state),*r.state);out["busy_rejected"]=nested_ok;out["status"]=int(r.report.status);out["solves"]=r.report.solve_count;
 }else throw std::runtime_error("case");out["ok"]=true;Json::StreamWriterBuilder b;b["indentation"]="";std::cout<<"RESULT "<<Json::writeString(b,out)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
