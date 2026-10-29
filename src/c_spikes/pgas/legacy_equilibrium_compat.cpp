#include "include/legacy_equilibrium_compat.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstddef>

// Pinned direct LP64 LAPACK provider with trailing Fortran character length.
extern "C" void dgesdd_(const char*,const int*,const int*,double*,
    const int*,double*,double*,const int*,double*,const int*,double*,
    const int*,int*,int*,std::size_t);
#ifdef CSP_G2AL_TEST_OBSERVER
extern "C" void csp_g2al_observe(std::size_t,double,const double*,double);
#endif
namespace csp_model {
State12 legacy_equilibrium12_compat(const Generator9& generator,
    double indicator_total, double stored_buffer_bound, double calcium) {
    static_assert(sizeof(int)==4,"qualified LP64 integer ABI required");
    static_assert(sizeof(std::size_t)==8,"qualified character-length ABI required");
    alignas(16) double a[81],u[81],vt[81],s[9],work[603];
    alignas(16) int iwork[72];
    for(std::size_t i=0;i<81;++i)a[i]=generator[i];
    for(double value:a)if(!std::isfinite(value))
        throw std::runtime_error("null(): svd failed");
    const char job='A';const int n=9,lwork=603;int info=0;
    dgesdd_(&job,&n,&n,a,&n,s,u,&n,vt,&n,work,&lwork,iwork,&info,std::size_t(1));
    if(info!=0)throw std::runtime_error("null(): svd failed");
    const double epsilon=std::numeric_limits<double>::epsilon();
    const double tolerance=(9*s[0])*epsilon;
    std::size_t rank=0;for(double value:s)if(value>tolerance)++rank;
    if(rank==9){
#ifdef CSP_G2AL_TEST_OBSERVER
        csp_g2al_observe(rank,tolerance,nullptr,0);
#endif
        throw std::out_of_range("Mat::col(): index out of bounds");
    }
    // V is transpose(VT); the first null column is V[:,rank].
    alignas(16) double selected[9];
    for(std::size_t i=0;i<9;++i){
        const double value=vt[rank+9*i];
        selected[i]=std::abs(value)<epsilon?double(0):value;
    }
    double acc1=0,acc2=0;
    for(std::size_t i=0;i<8;i+=2){acc1+=selected[i];acc2+=selected[i+1];}
    acc1+=selected[8];const double sum=acc1+acc2;
#ifdef CSP_G2AL_TEST_OBSERVER
    csp_g2al_observe(rank,tolerance,selected,sum);
#endif
    State12 result;
    for(std::size_t i=0;i<9;++i)result[i]=(selected[i]/sum)*indicator_total;
    result[9]=stored_buffer_bound;result[10]=calcium;result[11]=calcium;
    return result;
}
}
