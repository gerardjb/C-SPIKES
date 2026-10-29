#ifndef CSP_BRIGHT_INDICES5_ARMA_BRIDGE_H
#define CSP_BRIGHT_INDICES5_ARMA_BRIDGE_H
#include "bright_indices5.h"
#include <armadillo>
#include <limits>
namespace csp_model {
inline arma::uvec::fixed<5> owned_arma(const BrightIndices5& source) {
    static_assert(std::numeric_limits<arma::uword>::digits == 64,
                  "Brightness-index bridge requires the qualified 64-bit Armadillo index width");
    arma::uvec::fixed<5> result;
    for (std::size_t i=0;i<5;++i) result[i]=source[i];
    return result;
}
}
#endif
