#include <sstream>
#include <armadillo>
#include "include/legacy_equilibrium_compat.h"
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <array>
#include <cstdint>
#include <typeinfo>
using csp_model::Generator9;
using csp_model::State12;
namespace {
std::string id; int calls=0; bool inject=false; double factors[9],vectors[81];
std::array<double,9> selected{}; bool selected_valid=false; std::size_t observed_rank=99; double observed_tau=0,observed_sum=0;
using Backend=void(*)(const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*,int*,std::size_t);
void value(double v){if(std::isnan(v))std::cout<<"nan";else {std::uint64_t x;std::memcpy(&x,&v,8);std::cout<<std::hex<<x<<std::dec;}std::cout<<' ';}
}
extern "C" void __wrap_dgesdd_(const char* j,const int* m,const int* n,double* a,const int* lda,double* s,double* u,const int* ldu,double* vt,const int* ldvt,double* work,const int* lw,int* iw,int* info,std::size_t len){
 ++calls;
 if(*j!='A'||*m!=9||*n!=9||*lda!=9||*ldu!=9||*ldvt!=9||*lw!=603||len!=1)throw std::runtime_error("backend ABI mismatch");
 if(reinterpret_cast<std::uintptr_t>(a)%16 || reinterpret_cast<std::uintptr_t>(u)%16 || reinterpret_cast<std::uintptr_t>(vt)%16)throw std::runtime_error("alignment mismatch");
 if(inject){
  const double eps=std::numeric_limits<double>::epsilon();
  for(int c=0;c<9;++c){s[c]=c<8?double(9-c):0;for(int r=0;r<9;++r){u[r+9*c]=r==c?1:0;vt[r+9*c]=r==c?1:0;}}
  // Last V column is a controlled selected vector, encoded as VT row 8.
  for(int r=0;r<9;++r)vt[8+9*r]=double(r+1);
  if(id=="B01")vt[8]=std::nextafter(eps,0.0);
  if(id=="B02")vt[8]=eps;
  if(id=="B03")vt[8]=std::nextafter(eps,std::numeric_limits<double>::infinity());
  const double tau=(9*s[0])*eps;
  if(id=="B04")s[8]=std::nextafter(tau,0.0);
  if(id=="B05")s[8]=tau;
  if(id=="B06")s[8]=std::nextafter(tau,std::numeric_limits<double>::infinity());
  if(id=="B07"){s[7]=0;for(int r=0;r<9;++r)vt[7+9*r]=double(11+r);}
  if(id=="B08"){for(int r=0;r<9;++r)vt[8+9*r]=0;vt[8]=1;vt[8+9]=-1;}
  *info=id=="B12"?1:0;
 }else{
  auto real=reinterpret_cast<Backend>(dlsym(RTLD_NEXT,"wrapper2_dgesdd_"));if(!real)throw std::runtime_error("backend unavailable");
  real(j,m,n,a,lda,s,u,ldu,vt,ldvt,work,lw,iw,info,len);
 }
 for(int i=0;i<9;++i)factors[i]=s[i];for(int i=0;i<81;++i)vectors[i]=vt[i];
}
// Test-only observer emitted by candidate; absent in production compilation.
extern "C" void csp_g2al_observe(std::size_t rank,double tau,const double* v,double sum){
 observed_rank=rank;observed_tau=tau;observed_sum=sum;selected_valid=v!=nullptr;if(v)for(int i=0;i<9;++i)selected[i]=v[i];
}
int case_entry(int argc,char**argv){
 if(argc!=5)return 3;id=argv[1];std::string route=argv[2];double mass=std::stod(argv[4]);Generator9 g;g.fill(0);inject=id[0]=='B';
 if(!inject){std::ifstream f(std::string(argv[3])+"/"+id+".txt");std::string token;for(int r=0;r<9;++r)for(int c=0;c<9;++c){if(!(f>>token))return 4;g(r,c)=std::stod(token);}}
 if(id=="B09")mass=-1;if(id=="B10")mass=0;if(id=="B11")mass=std::numeric_limits<double>::quiet_NaN();
 std::string error="none",message;State12 out;bool success=false;
 try{out=csp_model::legacy_equilibrium12_compat(g,mass,0.125,1e-7);success=true;}
 catch(const std::out_of_range&e){error="out_of_range";message=e.what();}
 catch(const std::runtime_error&e){error="runtime_error";message=e.what();}
 std::cout<<"calls "<<calls<<" error "<<error<<" "<<message<<'\n';
 if(calls){std::cout<<"s ";for(double v:factors)value(v);std::cout<<"\nvt ";for(double v:vectors)value(v);std::cout<<'\n';}
 if(observed_rank!=99){std::cout<<"rank "<<observed_rank<<" tau ";value(observed_tau);std::cout<<" selected "<<selected_valid<<" ";if(selected_valid){for(double v:selected)value(v);std::cout<<"sum ";value(observed_sum);}std::cout<<'\n';}
 if(success){std::cout<<"result ";for(double v:out)value(v);std::cout<<'\n';}
 if(id=="A_NONFINITE" && (calls!=0||error!="runtime_error"))return 5;
 if(id=="B12" && (calls!=1||error!="runtime_error"))return 6;
 if(id=="B06" && error!="out_of_range")return 7;
 return 0;
}

int main(int argc,char**argv){if(argc!=1)return 90;for(const char* cid:{"B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12"}){calls=0;inject=false;selected.fill(0);selected_valid=false;observed_rank=99;observed_tau=observed_sum=0;std::ostringstream cap;auto* old=std::cout.rdbuf(cap.rdbuf());char* args[]={argv[0],const_cast<char*>(cid),const_cast<char*>("candidate"),const_cast<char*>("unused"),const_cast<char*>("0x1p+0")};int rc=case_entry(5,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<cid<<"\n"<<cap.str()<<"\nCASE_END\n";}return 0;}
