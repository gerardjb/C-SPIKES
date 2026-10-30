#include "include/model_evolution_compat.h"
#include <cmath>
#include <algorithm>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace csp_model { namespace workspace {
#ifdef CSP_G2AJ_TEST_OBSERVER
namespace {thread_local int observed_parallel=0,observed_team=1;}
int gaussian_test_parallel(){return observed_parallel;}
int gaussian_test_team(){return observed_team;}
#endif
// Expression order and paired/parallel evaluation adapted from Armadillo
// eop_core_meat.hpp and mp_misc.hpp (Apache-2.0). Retain std::pow and exp.
HistoryVector gaussian_input_compat(ScalarInputView grid,double deltat,double DCaT,double sigma2) {
    const double pi=3.14159265358979323846264338327950288419716939937510;
    const double amplitude=DCaT/std::sqrt(2*pi*sigma2);
    const double k=-0.5/sigma2,centre=deltat/2;
    HistoryVector out;out.set_size(grid.size());
    const auto n=grid.size();bool parallel=false;int threads=1;
#ifdef _OPENMP
    parallel=n>=320 && !omp_in_parallel();
    threads=std::min(8,std::max(1,omp_get_max_threads()));
#endif
#ifdef CSP_G2AJ_TEST_OBSERVER
    observed_parallel=parallel;int actual_team=1;
#endif
    auto inner=[&](std::size_t i){return std::exp(std::pow(centre-grid(i),double(2))*k);};
    if(parallel){
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
#endif
        // MSVC's OpenMP implementation requires a signed canonical loop index.
        for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(n);++i){
#ifdef CSP_G2AJ_TEST_OBSERVER
#ifdef _OPENMP
            if(i==0)actual_team=omp_get_num_threads();
#endif
#endif
            out(i)=inner(i)*amplitude;
        }
    }else{
        std::size_t i=0,j=1;
        for(;j<n;i+=2,j+=2){double a=inner(i),b=inner(j);a=a*amplitude;b=b*amplitude;out(i)=a;out(j)=b;}
        if(i<n)out(i)=inner(i)*amplitude;
    }
#ifdef CSP_G2AJ_TEST_OBSERVER
    observed_team=actual_team;
#endif
    return out;
}
HistoryVector zeroed_temporary_compat(std::size_t n) {
    HistoryVector out;out.set_size(n);for(std::size_t i=0;i<n;++i)out(i)=0.0;return out;
}
void copy_indicator_row_compat(HistoryMatrix& target,std::size_t row,const Indicator9& values) {
    for(std::size_t col=0;col<target.cols();++col)target(row,col)=values(col);
}
}}
