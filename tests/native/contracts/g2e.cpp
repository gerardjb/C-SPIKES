#include <armadillo>
#include <algorithm>
#include <sstream>
#include <memory>
#include "GCaMP_model.h"
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#ifdef G2E_CANDIDATE
using Matrix=csp_model::Generator9;
arma::mat::fixed<9,9> owned_armadillo(const csp_model::Generator9& source) {
    arma::mat::fixed<9,9> result;
    for (std::size_t col=0; col<9; ++col)
        for (std::size_t row=0; row<9; ++row)
            result(row,col)=source(row,col);
    return result;
}
#else
using Matrix=arma::mat::fixed<9,9>;
arma::mat::fixed<9,9> owned_armadillo(const Matrix& source){return source;}
#endif
void need(bool x,const char* s){if(!x)throw std::runtime_error(s);}
std::string hex(double x){if(std::isnan(x))return "nan";if(std::isinf(x))return x>0?"+inf":"-inf";char b[sizeof x];std::memcpy(b,&x,sizeof x);const char* h="0123456789abcdef";std::string s;for(unsigned char c:b){s+=h[c>>4];s+=h[c&15];}return s;}
template<class M>Json::Value values(const M& x){Json::Value a(Json::arrayValue);for(auto v:x){a.append(hex(v));}return a;}
Json::Value coords(const Matrix& m){Json::Value j(Json::arrayValue);for(int i=0;i<9;++i)for(int k=0;k<9;++k)j.append(hex(m(i,k)));return j;}
Json::Value cache(GCaMP& m){GCaMP_params p;m.read_params(p);Json::Value j;j.append(hex(p.konN));j.append(hex(p.konC));return j;}
int case_entry(int argc,char**argv){try{need(argc==5,"args");std::string id=argv[1];Json::Value out;out["id"]=id;
 if(id[0]=='M'||id[0]=='W'||id[0]=='A'){
  std::ifstream f(argv[2]);Matrix q;for(int i=0;i<9;++i)for(int j=0;j<9;++j){std::string token;need(bool(f>>token),"matrix input");std::size_t used=0;double x=std::stod(token,&used);need(used==token.size(),"matrix token");q(i,j)=x;}std::string extra;need(!(f>>extra),"trailing input");
  out["linear"]=values(q);out["coordinates"]=coords(q);const auto before=values(q);auto a=owned_armadillo(q);out["adapter"]=values(a);
  for(int j=0;j<9;++j){arma::vec::fixed<9> v;v.zeros();v(j)=1.;arma::vec product=a*v;if(id!="A_NONFINITE")need(product.is_finite(),"finite bridge fixture product");out["products"].append(values(product));}
  a(0,0)+=17.;need(values(q)==before,"adapter aliases canonical storage");
 }else{
  arma::vec gp,cp;need(gp.load(argv[3],arma::raw_ascii)&&cp.load(argv[4],arma::raw_ascii),"parameters");GCaMP m(csp_model::ScalarInputView{gp.memptr(),gp.n_elem},csp_model::ScalarInputView{cp.memptr(),cp.n_elem});Matrix local;
  m.setGmat(5e-8);out["cache_before"]=cache(m);
  if(id=="G01")m.setGmat(0.);else if(id=="G02")m.setGmat(5e-8);else if(id=="G03")m.setGmat(1e-2);
  else if(id=="G04"){m.fillGmat(local,5e-8);out["local"]=values(local);out["local_coordinates"]=coords(local);}
  else if(id=="G05"){
   for(int k=0;k<81;++k)m.Gmat[k]=(k+1)*.125;auto before=values(m.Gmat);auto prior=cache(m);m.setGmat_konN_konC(m.Gmat,2.25,3.5);need(cache(m)==prior,"partial changed scalar cache");
   const int changed[]={0,1,10,20,21,4,40,50,51,16,43};for(int k=0;k<81;++k){bool write=false;for(int x:changed)write=write||x==k;if(!write)need(hex(m.Gmat[k])==before[k].asString(),"partial touched forbidden entry");}
  }else if(id=="G06"){
   for(int k=0;k<81;++k)m.Gmat[k]=(k+1)*.125;auto original=values(m.Gmat);auto copy=m.Gmat;copy(0,0)=-7.;need(values(m.Gmat)==original,"copy aliases");out["owned_copy"]=values(copy);
   for(int i=0;i<9;++i)for(int j=0;j<9;++j)need(&m.Gmat(i,j)==&m.Gmat[i+9*j],"layout mismatch");
  }else need(false,"unknown case");
  out["linear"]=values(m.Gmat);out["coordinates"]=coords(m.Gmat);out["cache_after"]=cache(m);
 }
 std::cout<<"RESULT "<<out<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}

int main(int argc,char**argv){ if(argc!=4)return 90; for(const char* id:{"G01","G02","G03","G04","G05","G06"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[3],argv[1],argv[2]};int rc=case_entry(5,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
