#include <armadillo>
#include <sstream>
#include "GCaMP_model.h"
#include <json/json.h>
#include <cstddef>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
static_assert(std::is_standard_layout<GCaMP_params>::value,"standard layout");
static_assert(std::is_trivially_copyable<GCaMP_params>::value,"trivially copyable");
static_assert(std::is_same<decltype(GCaMP_params::Gparams),double[14]>::value,"array type");
void check(bool x,const char* s){if(!x)throw std::runtime_error(s);}
int case_entry(int argc,char**argv){try{
 check(argc==4,"arguments");std::string kind=argv[1];Json::Value out;
 if(kind=="layout"){
 out["size"]=Json::UInt64(sizeof(GCaMP_params));out["alignment"]=Json::UInt64(alignof(GCaMP_params));
 out["standard_layout"]=std::is_standard_layout<GCaMP_params>::value;out["trivial_copy"]=std::is_trivially_copyable<GCaMP_params>::value;
 out["copy_assignable"]=std::is_copy_assignable<GCaMP_params>::value;
 out["offsets"]["Gparams"]=Json::UInt64(offsetof(GCaMP_params,Gparams));
out["offsets"]["G_tot"]=Json::UInt64(offsetof(GCaMP_params,G_tot));
static_assert(std::is_same<decltype(GCaMP_params::G_tot),double>::value,"field type");
out["offsets"]["gamma"]=Json::UInt64(offsetof(GCaMP_params,gamma));
static_assert(std::is_same<decltype(GCaMP_params::gamma),double>::value,"field type");
out["offsets"]["DCaT"]=Json::UInt64(offsetof(GCaMP_params,DCaT));
static_assert(std::is_same<decltype(GCaMP_params::DCaT),double>::value,"field type");
out["offsets"]["Rf"]=Json::UInt64(offsetof(GCaMP_params,Rf));
static_assert(std::is_same<decltype(GCaMP_params::Rf),double>::value,"field type");
out["offsets"]["konN"]=Json::UInt64(offsetof(GCaMP_params,konN));
static_assert(std::is_same<decltype(GCaMP_params::konN),double>::value,"field type");
out["offsets"]["koffN"]=Json::UInt64(offsetof(GCaMP_params,koffN));
static_assert(std::is_same<decltype(GCaMP_params::koffN),double>::value,"field type");
out["offsets"]["konC"]=Json::UInt64(offsetof(GCaMP_params,konC));
static_assert(std::is_same<decltype(GCaMP_params::konC),double>::value,"field type");
out["offsets"]["koffC"]=Json::UInt64(offsetof(GCaMP_params,koffC));
static_assert(std::is_same<decltype(GCaMP_params::koffC),double>::value,"field type");
out["offsets"]["konPN"]=Json::UInt64(offsetof(GCaMP_params,konPN));
static_assert(std::is_same<decltype(GCaMP_params::konPN),double>::value,"field type");
out["offsets"]["koffPN"]=Json::UInt64(offsetof(GCaMP_params,koffPN));
static_assert(std::is_same<decltype(GCaMP_params::koffPN),double>::value,"field type");
out["offsets"]["konPN2"]=Json::UInt64(offsetof(GCaMP_params,konPN2));
static_assert(std::is_same<decltype(GCaMP_params::konPN2),double>::value,"field type");
out["offsets"]["koffPN2"]=Json::UInt64(offsetof(GCaMP_params,koffPN2));
static_assert(std::is_same<decltype(GCaMP_params::koffPN2),double>::value,"field type");
out["offsets"]["konPC"]=Json::UInt64(offsetof(GCaMP_params,konPC));
static_assert(std::is_same<decltype(GCaMP_params::konPC),double>::value,"field type");
out["offsets"]["koffPC"]=Json::UInt64(offsetof(GCaMP_params,koffPC));
static_assert(std::is_same<decltype(GCaMP_params::koffPC),double>::value,"field type");
out["offsets"]["konPC2"]=Json::UInt64(offsetof(GCaMP_params,konPC2));
static_assert(std::is_same<decltype(GCaMP_params::konPC2),double>::value,"field type");
out["offsets"]["koffPC2"]=Json::UInt64(offsetof(GCaMP_params,koffPC2));
static_assert(std::is_same<decltype(GCaMP_params::koffPC2),double>::value,"field type");
out["offsets"]["c0"]=Json::UInt64(offsetof(GCaMP_params,c0));
static_assert(std::is_same<decltype(GCaMP_params::c0),const double>::value,"field type");
out["offsets"]["FWHM"]=Json::UInt64(offsetof(GCaMP_params,FWHM));
static_assert(std::is_same<decltype(GCaMP_params::FWHM),const double>::value,"field type");
out["offsets"]["sigma2_calcium_spike"]=Json::UInt64(offsetof(GCaMP_params,sigma2_calcium_spike));
static_assert(std::is_same<decltype(GCaMP_params::sigma2_calcium_spike),double>::value,"field type");
out["offsets"]["gam_in"]=Json::UInt64(offsetof(GCaMP_params,gam_in));
static_assert(std::is_same<decltype(GCaMP_params::gam_in),double>::value,"field type");
out["offsets"]["gam_out"]=Json::UInt64(offsetof(GCaMP_params,gam_out));
static_assert(std::is_same<decltype(GCaMP_params::gam_out),double>::value,"field type");
out["offsets"]["koff_B"]=Json::UInt64(offsetof(GCaMP_params,koff_B));
static_assert(std::is_same<decltype(GCaMP_params::koff_B),const double>::value,"field type");
out["offsets"]["kon_B"]=Json::UInt64(offsetof(GCaMP_params,kon_B));
static_assert(std::is_same<decltype(GCaMP_params::kon_B),const double>::value,"field type");
out["offsets"]["B_tot"]=Json::UInt64(offsetof(GCaMP_params,B_tot));
static_assert(std::is_same<decltype(GCaMP_params::B_tot),const double>::value,"field type");
out["offsets"]["BCa0"]=Json::UInt64(offsetof(GCaMP_params,BCa0));
static_assert(std::is_same<decltype(GCaMP_params::BCa0),double>::value,"field type");
out["offsets"]["kapB"]=Json::UInt64(offsetof(GCaMP_params,kapB));
static_assert(std::is_same<decltype(GCaMP_params::kapB),double>::value,"field type");
out["offsets"]["csat"]=Json::UInt64(offsetof(GCaMP_params,csat));
static_assert(std::is_same<decltype(GCaMP_params::csat),const double>::value,"field type");
out["offsets"]["G0"]=Json::UInt64(offsetof(GCaMP_params,G0));
static_assert(std::is_same<decltype(GCaMP_params::G0),double>::value,"field type");
out["offsets"]["Gsat"]=Json::UInt64(offsetof(GCaMP_params,Gsat));
static_assert(std::is_same<decltype(GCaMP_params::Gsat),double>::value,"field type");
out["offsets"]["Ginit"]=Json::UInt64(offsetof(GCaMP_params,Ginit));
static_assert(std::is_same<decltype(GCaMP_params::Ginit),double>::value,"field type");
GCaMP_params p;
out["defaults"]["c0"]=p.c0;
out["defaults"]["FWHM"]=p.FWHM;
out["defaults"]["koff_B"]=p.koff_B;
out["defaults"]["kon_B"]=p.kon_B;
out["defaults"]["B_tot"]=p.B_tot;
out["defaults"]["csat"]=p.csat;
}else{check(kind=="handoff","case");arma::vec gp,cp;check(gp.load(argv[2],arma::raw_ascii)&&cp.load(argv[3],arma::raw_ascii),"fixture load");GCaMP model({gp.memptr(),gp.n_elem},{cp.memptr(),cp.n_elem});GCaMP_params p;model.read_params(p);
 for(double x:p.Gparams){check(std::isfinite(x),"Gparams finite");out["Gparams"].append(x);}
check(std::isfinite(p.G_tot),"G_tot finite");out["members"]["G_tot"]=p.G_tot;
check(std::isfinite(p.gamma),"gamma finite");out["members"]["gamma"]=p.gamma;
check(std::isfinite(p.DCaT),"DCaT finite");out["members"]["DCaT"]=p.DCaT;
check(std::isfinite(p.Rf),"Rf finite");out["members"]["Rf"]=p.Rf;
check(std::isfinite(p.konN),"konN finite");out["members"]["konN"]=p.konN;
check(std::isfinite(p.koffN),"koffN finite");out["members"]["koffN"]=p.koffN;
check(std::isfinite(p.konC),"konC finite");out["members"]["konC"]=p.konC;
check(std::isfinite(p.koffC),"koffC finite");out["members"]["koffC"]=p.koffC;
check(std::isfinite(p.konPN),"konPN finite");out["members"]["konPN"]=p.konPN;
check(std::isfinite(p.koffPN),"koffPN finite");out["members"]["koffPN"]=p.koffPN;
check(std::isfinite(p.konPN2),"konPN2 finite");out["members"]["konPN2"]=p.konPN2;
check(std::isfinite(p.koffPN2),"koffPN2 finite");out["members"]["koffPN2"]=p.koffPN2;
check(std::isfinite(p.konPC),"konPC finite");out["members"]["konPC"]=p.konPC;
check(std::isfinite(p.koffPC),"koffPC finite");out["members"]["koffPC"]=p.koffPC;
check(std::isfinite(p.konPC2),"konPC2 finite");out["members"]["konPC2"]=p.konPC2;
check(std::isfinite(p.koffPC2),"koffPC2 finite");out["members"]["koffPC2"]=p.koffPC2;
check(std::isfinite(p.c0),"c0 finite");out["members"]["c0"]=p.c0;
check(std::isfinite(p.FWHM),"FWHM finite");out["members"]["FWHM"]=p.FWHM;
check(std::isfinite(p.sigma2_calcium_spike),"sigma2_calcium_spike finite");out["members"]["sigma2_calcium_spike"]=p.sigma2_calcium_spike;
check(std::isfinite(p.gam_in),"gam_in finite");out["members"]["gam_in"]=p.gam_in;
check(std::isfinite(p.gam_out),"gam_out finite");out["members"]["gam_out"]=p.gam_out;
check(std::isfinite(p.koff_B),"koff_B finite");out["members"]["koff_B"]=p.koff_B;
check(std::isfinite(p.kon_B),"kon_B finite");out["members"]["kon_B"]=p.kon_B;
check(std::isfinite(p.B_tot),"B_tot finite");out["members"]["B_tot"]=p.B_tot;
check(std::isfinite(p.BCa0),"BCa0 finite");out["members"]["BCa0"]=p.BCa0;
check(std::isfinite(p.kapB),"kapB finite");out["members"]["kapB"]=p.kapB;
check(std::isfinite(p.csat),"csat finite");out["members"]["csat"]=p.csat;
check(std::isfinite(p.G0),"G0 finite");out["members"]["G0"]=p.G0;
check(std::isfinite(p.Gsat),"Gsat finite");out["members"]["Gsat"]=p.Gsat;
check(std::isfinite(p.Ginit),"Ginit finite");out["members"]["Ginit"]=p.Ginit;
}
 Json::StreamWriterBuilder w;w["indentation"]="";w["precision"]=17;std::cout<<"RESULT "<<Json::writeString(w,out)<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}

int main(int argc,char**argv){if(argc!=3)return 90;for(const char* id:{"layout","handoff"}){std::ostringstream captured;auto* old=std::cout.rdbuf(captured.rdbuf());char* args[]={argv[0],const_cast<char*>(id),argv[1],argv[2]};int rc=case_entry(4,args);std::cout.rdbuf(old);if(rc)return rc;std::cout<<"CASE_BEGIN "<<id<<"\n"<<captured.str()<<"\nCASE_END\n";}return 0;}
