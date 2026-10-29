#include "include/GCaMP_model.h"
#include "../observer.h"
#include <json/json.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <cstdlib>
std::string bytes(const std::vector<double>& v){const auto* p=reinterpret_cast<const unsigned char*>(v.data());const char*h="0123456789abcdef";std::string s;for(size_t i=0;i<v.size()*8;++i){s+=h[p[i]>>4];s+=h[p[i]&15];}return s;}
Json::Value packed(const std::vector<double>& v,size_t rows,size_t cols=0){Json::Value x;x["shape"].append(Json::UInt64(rows));if(cols)x["shape"].append(Json::UInt64(cols));x["dtype"]="<f8";x["bytes"]=bytes(v);return x;}
std::string exact(double x){std::ostringstream s;s<<std::hexfloat<<x;return s.str();}
void emit(const char* tag,const Json::Value& x){Json::StreamWriterBuilder w;w["indentation"]="";w["precision"]=17;std::cout<<tag<<Json::writeString(w,x)<<std::endl;}
void require(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
int main(int argc,char**argv){try{
 require(argc==5,"arguments");std::string id=argv[1];
 Json::Value plan;std::ifstream pf(argv[4]);pf>>plan;Json::Value fixture;for(const auto& row:plan["cases"])if(row["id"].asString()==id)fixture=row;require(!fixture.isNull(),"fixture");
 auto gp=i2b::read_params(argv[2]);auto cp=i2b::read_params(argv[3]);
 GCaMP m({gp.data(),gp.size()},{cp.data(),cp.size()});m.setTimeStepMode(FIXEDLA);auto initial=m.steady_state(5e-8);m.init();
 std::vector<double> times(4),spikes(fixture["spikes_hex"].size());for(size_t i=0;i<4;++i)times[i]=std::strtod(fixture["time_hex"][Json::ArrayIndex(i)].asCString(),nullptr);for(size_t i=0;i<spikes.size();++i)spikes[i]=std::strtod(fixture["spikes_hex"][Json::ArrayIndex(i)].asCString(),nullptr);
 m.integrateOverTime({times.data(),times.size()},{spikes.data(),spikes.size()});const auto& s=l3e::snapshot();require(s.calls==1,"capture count");
 auto G=m.getGValues();auto B=m.getBCaValues();auto C=m.getCaValues();auto I=m.getCaInValues();require(G.rows()==4&&G.cols()==9&&B.size()==4&&C.size()==4&&I.size()==4,"output shapes");
 std::vector<double> out;for(size_t i=0;i<4;++i){for(size_t j=0;j<9;++j)out.push_back(G(i,j));out.push_back(B(i));out.push_back(C(i));out.push_back(I(i));}
 Json::Value value;value["time"]=packed(std::vector<double>(times.begin(),times.end()),4);value["state"]=packed(out,4,12);
 for(int i=0;i<9;++i)value["order"].append("G"+std::to_string(i));for(const char* name:{"BCa","Ca","Ca_in"})value["order"].append(name);
 value["buffer"]="linear";value["method"]="integrateOverTime";value["dt"]=1e-4;
 for(size_t col=0;col<12;++col){unsigned count=0;for(size_t row=0;row<4;++row)count+=std::isfinite(out[12*row+col]);value["finite_counts"].append(count);}
 Json::Value diag;diag["id"]=id;diag["M"]=s.M;diag["fine_dt"]=s.fine_dt;diag["fine_dt_hex"]=exact(s.fine_dt);diag["new_t_end"]=s.new_t_end;diag["new_t_end_hex"]=exact(s.new_t_end);
 diag["requested"]=packed(s.time,s.time.size());diag["grid"]=packed(s.grid,s.grid.size());diag["pre_states"]=packed(s.states,s.grid.size(),12);diag["spikes"]=packed(s.spikes,s.spikes.size());diag["calcium_input"]=packed(s.input,s.input.size());
 for(double x:s.grid)diag["grid_hex"].append(exact(x));for(double x:s.time)diag["requested_hex"].append(exact(x));
 emit("DIAGNOSTIC ",diag);emit("RESULT ",value);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}}
