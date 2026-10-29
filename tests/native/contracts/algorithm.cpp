#include <armadillo>
#include <sstream>
#include <armadillo>
#include "model_workspace_arma_bridge.h"
#include "model_algorithm_compat.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <array>
#ifndef ARMA_CHECK_CONFORMANCE
#error conformance required
#endif
#ifdef __FAST_MATH__
#error reassociation excluded
#endif
static_assert(ARMA_WARN_LEVEL==2,"pinned warning level");
void emit(double x){if(std::isnan(x)){std::cout<<"nan ";return;}unsigned long long v;std::memcpy(&v,&x,8);std::cout<<std::hex<<v<<' ';}
void need(bool x){if(!x)throw std::runtime_error("probe assertion");}
using V=std::vector<double>;
const double inf=std::numeric_limits<double>::infinity(),probe_nan=std::numeric_limits<double>::quiet_NaN();
csp_model::ScalarInputView view(const V& v){return {v.data(),v.size()};}
void interpolation(bool candidate,V x,V y,V q,int alias=0){
 const V ox=x,oy=y,oq=q;V result{42.,-0.};std::string error;
 try{
  if(candidate){auto v=csp_model::linear_interpolate_compat(view(x),view(y),view(q));V tmp(v.size());for(size_t i=0;i<v.size();++i)tmp[i]=v(i);if(alias==1)x=tmp;else if(alias==2)y=tmp;else if(alias==3)q=tmp;result=tmp;}
  else{arma::vec a(x),b(y),c(q),r(result);if(alias==1){arma::interp1(a,b,c,a);r=a;}else if(alias==2){arma::interp1(a,b,c,b);r=b;}else if(alias==3){arma::interp1(a,b,c,c);r=c;}else arma::interp1(a,b,c,r);result.assign(r.begin(),r.end());if(alias==1)x=result;else if(alias==2)y=result;else if(alias==3)q=result;}
 }catch(const std::logic_error&e){error=std::string("logic_error:")+e.what();}
 std::cout<<error<<'\n';for(double v:result)emit(v);std::cout<<'\n';for(auto *v:{&x,&y,&q}){for(double a:*v)emit(a);std::cout<<'\n';}
 if(error.size()||!alias)need(x==ox || std::memcmp(x.data(),ox.data(),x.size()*8)==0);
}
template<size_t N>void parse(bool candidate,const std::string&path){bool ok=false;std::string error;std::array<double,N> out;out.fill(-99.);
 try{if(candidate){csp_model::ParameterBlock<N> p;p.fill(-99.);ok=csp_model::load_raw_ascii_compat(p,path);std::copy(p.begin(),p.end(),out.begin());}else{typename arma::Col<double>::template fixed<N> p;p.fill(-99.);ok=p.load(path,arma::raw_ascii);std::copy(p.begin(),p.end(),out.begin());}}
 catch(const std::exception&e){error=e.what();}
 std::cout<<ok<<' '<<error<<'\n';for(auto v:out)emit(v);std::cout<<'\n';}
int case_entry(int argc,char**argv){try{need(argc==4);std::string id=argv[1],route=argv[2],root=argv[3];bool candidate=route=="candidate";need(candidate||route=="baseline");int k=std::stoi(id.substr(1));
 if(id[0]=='A'){
 csp_model::Generator9 a;csp_model::workspace::Indicator9 g,d;a.fill(0.);for(unsigned i=0;i<9;++i){g[i]=(i+1)*.125;a(i,i)=double(i+1);}
 if(k==2){std::ifstream f(root+"/M01.txt");for(unsigned r=0;r<9;++r)for(unsigned c=0;c<9;++c)need(bool(f>>a(r,c)));}
 if(k==3||k==4)for(unsigned r=0;r<9;++r)for(unsigned c=0;c<9;++c)a(r,c)=k==3?double(int((r*11+c*7)%19)-9)/8.:((c%3==0)?1e16:(c%3==1?1.:-1e16));
 if(k==5){for(unsigned i=0;i<9;++i)g[i]=-0.;a.fill(-0.);}if(k==6)for(unsigned i=0;i<9;++i)g[i]=(i%2?-1.:1.)*std::numeric_limits<double>::denorm_min();if(k==7)for(unsigned i=0;i<9;++i)g[i]=std::numeric_limits<double>::max();if(k==8){g[0]=inf;g[4]=probe_nan;}
 auto original=g;auto matrix=a;csp_model::workspace::product9_compat(a,g,d);for(double v:d.values)emit(v);std::cout<<'\n';
 if(candidate)csp_model::workspace::advance9_compat(g,.0001,d);else{arma::vec v(g.data(),9),dv(d.data(),9);v=v+.0001*dv;for(unsigned i=0;i<9;++i)g[i]=v[i];}
 for(double v:g.values)emit(v);std::cout<<'\n';for(double v:original.values)emit(v);for(double v:a.coefficients)emit(v);std::cout<<'\n';need(std::memcmp(matrix.data(),a.data(),648)==0);
 }else if(id[0]=='I'){
 switch(k){
 case 1:interpolation(candidate,{0,1,2,3},{2,5,-4,8},{.25,.75,1.5,2.25});break;
 case 2:interpolation(candidate,{0,1,2},{1,4,9},{1.7,.2,1.7,0.,2.,.2});break;
 case 3:interpolation(candidate,{0,1,2},{1,probe_nan,3},{0,std::nextafter(0.,1.),1,std::nextafter(1.,0.),std::nextafter(1.,2.),2});break;
 case 4:interpolation(candidate,{0,1,2},{1,2,4},{-.1,0.,2.,2.1});break;
 case 5:interpolation(candidate,{2,0,1,1,0,2},{20,0,10,11,1,21},{0,.5,1,1.5,2});break;
 case 6:for(int n:{15,16,17,31,32,33}){V x(n),y(n);for(int i=0;i<n;++i){x[i]=(i*7)%5;y[i]=i+.125;}interpolation(candidate,x,y,{0,.5,1,2.5,4});}break;
 case 7:interpolation(candidate,{-0.,0.,1.,2.},{-0.,0.,-0.,0.},{-0.,0.,.5,1.,2.});break;
 case 8:interpolation(candidate,{-inf,-inf,0,1,inf,inf},{1,2,3,4,5,6},{-inf,-1,0,.5,1,2,inf});break;
 case 9:interpolation(candidate,{0,probe_nan,1},{1,2,3},{0,.5,1});break;
 case 10:interpolation(candidate,{0,1,2},{1,2,4},{0,probe_nan,2});interpolation(candidate,{0,1,2},{1,2,4},{2,0,probe_nan});break;
 case 11:interpolation(candidate,{},{},{});interpolation(candidate,{0,1},{1},{0});interpolation(candidate,{0},{1},{0});interpolation(candidate,{1,1},{1,2},{});interpolation(candidate,{0,1},{1,2},{});break;
 case 12:for(int a:{0,1,2,3}){interpolation(candidate,{0,1,2},{1,3,7},{1.5,.5},a);interpolation(candidate,{0,probe_nan,2},{1,3,7},{1.5,.5},a);}break;
 default:need(false);}
 }else if(id[0]=='P'){
 for(int n:{6,14})for(int sub=0;sub<3;++sub){auto path=root+"/P"+(k<10?std::string("0"):std::string())+std::to_string(k)+"_"+std::to_string(n)+"_"+std::to_string(sub)+".txt";if(n==6)parse<6>(candidate,path);else parse<14>(candidate,path);}
 }else if(id[0]=='M'){
 for(int sub=0;sub<3;++sub){std::array<double,100> a;for(size_t i=0;i<100;++i)a[i]=k==1?(sub==0?double(i):(sub==1?-double(i):double(int(i*13%17)-8))):(k==2?(i%2?-0.:0.):(k==3?probe_nan:-inf));
 if(k==2&&sub==1)for(size_t i=0;i<100;++i)a[i]=i%2?0.:-0.;if(k==2&&sub==2){a.fill(-1);a[20]=a[31]=2.;}
 if(k==3&&sub==1)a[71]=7.;if(k==3&&sub==2){a[0]=-inf;a[99]=inf;}if(k==4&&sub==1)a[98]=inf;if(k==4&&sub==2){a[1]=inf;a[98]=probe_nan;}
 auto before=a;double result;if(candidate)result=csp_model::max100_compat(a);else{arma::vec v(a.data(),100);result=v.max();}emit(result);std::cout<<'\n';need(std::memcmp(before.data(),a.data(),800)==0);}
 }else need(false);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}

int main(int argc,char**argv){if(argc!=4)return 90;for(const char* id:{"A01","A02","A03","A04","A05","A06","A07","A08","I01","I02","I03","I04","I05","I06","I07","I08","I09","I10","I11","I12","P01","P02","P03","P04","P05","P06","P07","P08","M01","M02","M03","M04"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),const_cast<char*>("candidate"),argv[3]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
