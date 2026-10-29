#include "include/model_backend.h"
#include <cstddef>
#include <climits>
// Qualified direct LP64 Fortran provider, including the hidden character length.
// Provider selection is pinned in cmake/PinnedNumericalProvider.cmake.
static_assert(sizeof(int)==4 && sizeof(std::size_t)==8 && CHAR_BIT==8,"qualified BLAS ABI required");
extern "C" void dgemv_(const char*,const int*,const int*,const double*,const double*,const int*,const double*,const int*,const double*,double*,const int*,std::size_t);
namespace csp_model { namespace workspace {
void product9_compat(const Generator9& generator,const Indicator9& input,Indicator9& output) {
    alignas(16) double matrix[81];
    for(std::size_t col=0;col<9;++col)for(std::size_t row=0;row<9;++row)matrix[row+9*col]=generator(row,col);
    const char trans='N';const int n=9,inc=1;const double alpha=1.0,beta=0.0;
    dgemv_(&trans,&n,&n,&alpha,matrix,&n,input.data(),&inc,&beta,output.data(),&inc,std::size_t(1));
}
}}
