// Frozen F01-F08 contract tests. No production model is linked.
#include "checked_equilibrium.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using namespace checked::detail;
namespace {
int calls=0;
Factors fail_backend(const Matrix9&) { ++calls; Factors f; f.code=77; f.s.fill(NAN); return f; }
void require(bool ok) { if (!ok) throw std::runtime_error("contract assertion failed"); }
Factors identity() { Factors f; f.code=0; f.s.fill(1); for(int i=0;i<9;++i) f.u[9*i+i]=f.v[9*i+i]=1; return f; }
}
int contract_case(const std::string& id) {
 Matrix9 q{}; for(int i=0;i<9;++i)q[9*i+i]=1;
 if(id=="F01") {
  Vector9 s{{1,.5,.25,.125,.0625,.03125,.015625,9*std::numeric_limits<double>::epsilon(),0}};
  require(classify(s)==std::array<int,3>{{8,7,7}});
 } else if(id=="F02") {
  auto r=evaluate(q,true,1,fail_backend);
  require(calls==1 && r.backend_called && r.factors.code==77 && r.outcome==Outcome::backend_failed && !r.has_diagnostics && !r.has_state);
 } else if(id=="F03") {
  for(int which=0;which<3;++which) {
   auto f=identity(); if(which==0)f.s[0]=NAN;else if(which==1)f.u[0]=NAN;else f.v[0]=NAN;
   auto r=assess(q,f,true,1); require(r.outcome==Outcome::malformed_factors && !r.has_diagnostics && !r.has_state);
  }
 } else if(id=="F04") {
  auto f=identity();f.s[0]=.5;auto r=assess(q,f,true,1);
  require(r.outcome==Outcome::malformed_factors && !r.has_state);
 } else if(id=="F05" || id=="F06") {
  auto r=evaluate(q,true,id=="F05"?0:NAN,fail_backend);
  require(calls==0 && !r.backend_called && r.outcome==Outcome::input_invalid && !r.has_state);
 } else if(id=="F07") {
  Vector9 z{};z[0]=1/std::sqrt(2.);z[1]=-z[0];Vector9 g;g.fill(42);const auto before=g;
  double sum=0,guard=0;require(!normalize(z,1,g,sum,guard));require(g==before && sum==0 && guard>0);
 } else if(id=="F08") {
  Vector9 g{};g[0]=-2e-12;g[1]=1+2e-12;auto before=g;
  require(!positive(g,1) && g==before);
 } else throw std::runtime_error("unknown contract case");
 std::cout<<"{\"case\":\""<<id<<"\",\"passed\":true}\n";return 0;
}

int main(){for(const auto& id:{"F01","F02","F03","F04","F05","F06","F07","F08"}){calls=0;contract_case(id);}return 0;}
