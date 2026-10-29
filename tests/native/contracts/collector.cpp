#include <armadillo>
#include "include/GCaMP_model.h"
#include "include/numerical_diagnostics.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace csp_diagnostics;
static Record last;
static unsigned records=0;
static void capture(const Record& r){last=r;++records;}
static void broken(const Record&){throw std::runtime_error("handler");}
static void require(bool v){if(!v)throw std::runtime_error("diagnostic assertion failed");}
int case_entry(int argc,char** argv){
 if(argc!=4)return 64;
 const std::string id(argv[1]);arma::vec gp,cp;gp.load(argv[2],arma::raw_ascii);cp.load(argv[3],arma::raw_ascii);
 if(id=="P06"){
  {Scope run("repeated-model-run",capture);GCaMP g({gp.memptr(),gp.n_elem},{cp.memptr(),cp.n_elem});for(int i=0;i<3;++i){g.setState(1e-6);g.fixedStep(.001,1);}}
  require(records==1 && last.detections>1 && last.first.mode==-1 && std::strcmp(last.first.integrator,"not_applicable")==0 && !last.first.timing_known && last.first.requested_dt==0 && last.first.h==0 && last.first.substep==-1 && std::strcmp(last.first.method,"GCaMP.setGmat")==0 && std::strcmp(last.first.field,"generator")==0 && last.complete);
  char a[1024],b[768],c[512];format(last,a,sizeof(a),b,sizeof(b),c,sizeof(c));
  require(std::strstr(a,"nominal_h=") && std::strstr(b,"suppressed="));
  std::cerr<<"WARNING "<<a<<"\nWARNING "<<b<<"\n";
 }else if(id=="P07"){
  {Scope s("finite",capture);s.scalar(0,"zero-calcium",rate_value);s.scalar(1,"finite",state_value);}require(records==0);
  {Scope s("invalid-values",capture);s.scalar(std::numeric_limits<double>::quiet_NaN(),"Ca",state_value);s.scalar(0,"denominator",normalization,-1,true);}
  require(records==1 && last.detections==2 && last.counts[normalization]==1);
  stderr_sink(last);
 }else if(id=="P14"){
  const auto start=std::chrono::steady_clock::now();
  {Scope run("bounded-valid-timing",capture);GCaMP g({gp.memptr(),gp.n_elem},{cp.memptr(),cp.n_elem});for(int i=0;i<64;++i){g.init();g.evolve(.001,1);require(std::isfinite(g.getDFF()));}}
  require(records==0);
  std::cout<<"elapsed_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" calls=64 comparable_baseline=unavailable\n";
 }else if(id=="P15"){
  bool caught=false;
  try{Scope s("interrupted",capture);s.scalar(std::numeric_limits<double>::infinity(),"output",output_value);throw std::runtime_error("original");}
  catch(const std::runtime_error& e){caught=std::string(e.what())=="original";}
  require(caught && records==1 && !last.complete);
  {Scope s("broken-sink",broken);s.scalar(std::numeric_limits<double>::infinity(),"output",output_value);}
 }else return 65;
 return 0;
}

int main(int argc,char**argv){if(argc!=3)return 90;for(const char* id:{"P06","P07","P14","P15"}){records=0;last=Record{};char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2]};int rc=case_entry(4,args);if(rc)return rc;}return 0;}
