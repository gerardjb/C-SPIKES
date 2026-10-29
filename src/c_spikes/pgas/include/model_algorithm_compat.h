#ifndef CSP_MODEL_ALGORITHM_COMPAT_H
#define CSP_MODEL_ALGORITHM_COMPAT_H
#include "history_storage.h"
#include "scalar_input_view.h"
#include "parameter_block.h"
#include <array>
#include <string>
namespace csp_model {
// Private owned results; borrowed inputs are read only during this call.
HistoryVector linear_interpolate_compat(ScalarInputView x,ScalarInputView y,ScalarInputView query);
double max100_compat(const std::array<double,100>& values);
namespace detail {
bool load_raw_ascii_values(double* destination,std::size_t size,const std::string& filename);
}
template<std::size_t N> bool load_raw_ascii_compat(ParameterBlock<N>& dst,const std::string& filename) {
    static_assert(N==6 || N==14,"only pinned parameter records are supported");
    // Match the former fresh fixed-vector temporary. Do not publish on exceptions.
    ParameterBlock<N> temporary;
    const bool loaded=detail::load_raw_ascii_values(temporary.data(),N,filename);
    dst=temporary;
    return loaded;
}
}
#endif
