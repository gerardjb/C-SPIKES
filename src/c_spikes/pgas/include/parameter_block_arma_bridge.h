#ifndef CSP_PARAMETER_BLOCK_ARMA_BRIDGE_H
#define CSP_PARAMETER_BLOCK_ARMA_BRIDGE_H
#include "parameter_block.h"
#include <armadillo>
#include <string>

namespace csp_model {
// Temporary owned storage preserves fixed-size assignment checks without aliases.
template<std::size_t N>
void assign_from_armadillo(ParameterBlock<N>& dst, const arma::vec& src) {
    typename arma::Col<double>::template fixed<N> temporary;
    temporary=src;
    for(std::size_t i=0;i<N;++i) dst[i]=temporary[i];
}
// Preserve the pinned raw_ascii parser and its normal false-return zeroing.
// Production uses this only at the existing fresh-constructor loading sites.
template<std::size_t N>
bool load_raw_ascii(ParameterBlock<N>& dst, const std::string& filename) {
    typename arma::Col<double>::template fixed<N> temporary;
    const bool loaded=temporary.load(filename,arma::raw_ascii);
    for(std::size_t i=0;i<N;++i) dst[i]=temporary[i];
    return loaded;
}
}
#endif
