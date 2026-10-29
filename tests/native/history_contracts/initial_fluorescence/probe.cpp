#include <algorithm>
#include "include/GCaMP_model.h"
#include "../observer.h"
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdlib>
void req(bool b,const char* m){if(!b)throw std::runtime_error(m);}
std::string bytes(const std::vector<double>& v){const unsigned char*p=reinterpret_cast<const unsigned char*>(v.data());const char*h="0123456789abcdef";std::string o;for(size_t i=0;i<v.size()*8;++i){o+=h[p[i]>>4];o+=h[p[i]&15];}return o;}
std::string bytes(const csp_model::State12& a){return bytes(std::vector<double>(a.begin(),a.end()));}
template<class V> std::string bytes(const V& a){std::vector<double> v;for(size_t i=0;i<a.size();++i)v.push_back(a(i));return bytes(v);}
std::vector<double> states(GCaMP& m){auto g=m.getGValues();auto b=m.getBCaValues();auto c=m.getCaValues();auto in=m.getCaInValues();std::vector<double> a;for(size_t r=0;r<g.rows();++r){for(size_t k=0;k<9;++k)a.push_back(g(r,k));a.push_back(b(r));a.push_back(c(r));a.push_back(in(r));}return a;}
double expected(GCaMP& m,const csp_model::State12& s){GCaMP_params par;m.read_params(par);double sum=0;for(int k:{2,3,5,6,8})sum+=s[k];return (sum-par.Ginit)/(par.Ginit-par.G0+(par.Gsat-par.G0)/(par.Rf-1));}
void check(double a,double b,double scale=0){if(!std::isfinite(b)){req(!std::isfinite(a),"invalid normalization hidden");return;}double bound=32*std::numeric_limits<double>::epsilon()*std::max({std::abs(a),std::abs(b),scale,std::numeric_limits<double>::min()});req(std::abs(a-b)<=bound,"initial fluorescence formula");}
int main(int argc,char**argv){try{
 req(argc==6,"arguments");std::string id=argv[1],variant=argv[5];Json::Value plan;std::ifstream p(argv[4]);p>>plan;Json::Value f;for(const auto& x:plan["cases"])if(x["id"].asString()==id)f=x;
 bool contract=id[0]=='C';if(contract)f=plan["cases"][1];req(!f.isNull(),"fixture");
 auto gp=i2b::read_params(argv[2]);auto cp=i2b::read_params(argv[3]);GCaMP m({gp.data(),gp.size()},{cp.data(),cp.size()});m.setTimeStepMode(FIXEDLA);m.setState(std::strtod(f["initial_ca_hex"].asCString(),nullptr));
 auto initial=m.state;const double initial_expected=expected(m,initial);
 std::vector<double> t(f["time_hex"].size()),sp(f["spikes_hex"].size());for(size_t i=0;i<t.size();++i)t[i]=std::strtod(f["time_hex"][Json::ArrayIndex(i)].asCString(),nullptr);for(size_t i=0;i<sp.size();++i)sp[i]=std::strtod(f["spikes_hex"][Json::ArrayIndex(i)].asCString(),nullptr);
 if(id=="C04"){GCaMP_params par;m.read_params(par);m.setParams(0.0,par.gamma,par.DCaT,par.Rf,par.gam_in,par.gam_out);}
 const double target=expected(m,initial);GCaMP_params par;m.read_params(par);double magnitude=std::abs(par.Ginit);for(int k:{2,3,5,6,8})magnitude+=std::abs(initial[k]);const double formula_scale=magnitude/std::abs(par.Ginit-par.G0+(par.Gsat-par.G0)/(par.Rf-1));m.integrateOverTime({t.data(),t.size()},{sp.data(),sp.size()});auto ff=m.getDFFValues();auto ss=states(m);const auto snap=l3f::snapshot();
 req(ff.size()==t.size()&&ss.size()==12*t.size(),"shapes");req(bytes(m.state)==bytes(initial),"stored state changed");
 if(variant=="candidate"){check(ff(0),target,formula_scale);if(std::isfinite(target))for(double x:i2b::copy(ff))req(std::isfinite(x),"nonfinite ordinary output");}
 if(contract){
  req(variant=="candidate","contract variant");
  if(id=="C01"){auto first=bytes(ff);m.setState(std::strtod(f["initial_ca_hex"].asCString(),nullptr));m.integrateOverTime({t.data(),t.size()},{sp.data(),sp.size()});req(first==bytes(m.getDFFValues())&&bytes(ss)==bytes(states(m)),"repeatability");}
  if(id=="C02"){auto first=bytes(ff);for(size_t i=0;i<ff.size();++i)ff(i)=123;req(first==bytes(m.getDFFValues()),"owned fluorescence result");}
  if(id=="C03"){req(initial_expected!=0,"nonresting control degenerate");check(ff(0),initial_expected,formula_scale);}
  if(id=="C04")req(!std::isfinite(ff(0)),"invalid denominator became finite");
 }
 Json::Value v;v["id"]=id;v["variant"]=variant;v["time"]=bytes(t);v["state"]=bytes(ss);v["stored_state"]=bytes(m.state);v["grid"]=bytes(snap.grid);v["input"]=bytes(snap.input);v["pre_states"]=bytes(snap.states);v["pre_fluorescence_tail"]=bytes(snap.tail);
 // Baseline first interval is undefined historical evidence, not a parity target.
 std::vector<double> stable;for(size_t i=0;i<t.size();++i)if(t[i]>=snap.grid[1])stable.push_back(m.getDFFValues()(i));v["later_fluorescence"]=bytes(stable);
 if(variant=="candidate")v["fluorescence"]=bytes(m.getDFFValues());v["initial_formula"]=bytes(std::vector<double>{target});v["formula_scale"]=bytes(std::vector<double>{formula_scale});
 Json::StreamWriterBuilder w;w["indentation"]="";std::cout<<"RESULT "<<Json::writeString(w,v)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
