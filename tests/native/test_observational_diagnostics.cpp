#include "numerical_diagnostics.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace csp_diagnostics;
static Record last; static unsigned records=0;
static void capture(const Record& r){last=r;++records;}
static void need(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
int main(){
 int evaluated=0;
 CSP_OBSERVE(++evaluated;)
 need(evaluated==(enabled?1:0),"OFF observation argument evaluated");
 {Scope s("finite",capture);s.scalar(1,"x",state_value);}
 need(records==0,"finite report");
 {Scope s("outer",capture);s.step(1,"FIXEDLA",.002,.0001,19,1e-6,2e-6,3e-6);
  {Scope inner("matrix");inner.equilibrium_context();inner.scalar(std::numeric_limits<double>::quiet_NaN(),"generator",rate_value,2);}
  if(enabled)need(active()->record.current.substep==19,"outer context");
  s.scalar(std::numeric_limits<double>::infinity(),"state",state_value);}
 if(enabled){need(records==1&&last.detections==2,"aggregation");need(std::strcmp(last.first.method,"matrix")==0&&!last.first.timing_known,"first context");char a[1024],b[768],c[512];format(last,a,sizeof(a),b,sizeof(b),c,sizeof(c));need(std::strstr(b,"suppressed=1"),"summary");}
 else need(records==0,"OFF report");
 try{Scope s("unwind",capture);s.scalar(std::numeric_limits<double>::infinity(),"bad",output_value);throw std::runtime_error("original");}catch(const std::runtime_error& e){need(std::strcmp(e.what(),"original")==0,"exception changed");}
 if(enabled)need(records==2&&!last.complete&&last.id==2,"completion and ID contract");
 std::cout<<"HEADER_MODE_PASS "<<enabled<<'\n';
}
