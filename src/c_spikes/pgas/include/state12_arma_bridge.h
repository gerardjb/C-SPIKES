#ifndef CSP_STATE12_ARMA_BRIDGE_H
#define CSP_STATE12_ARMA_BRIDGE_H
#include "state12.h"
#include <armadillo>
namespace csp_model {
inline arma::vec::fixed<12> owned_arma(const State12& source) {
    arma::vec::fixed<12> result;
    for (std::size_t i=0;i<12;++i) result[i]=source[i];
    return result;
}
inline arma::vec::fixed<9> owned_indicator(const State12& source) {
    arma::vec::fixed<9> result;
    for (std::size_t i=0;i<9;++i) result[i]=source[i];
    return result;
}
inline void assign_state(State12& destination,const arma::vec& source) {
    // Preserve Armadillo's fixed-size assignment checks before publication.
    arma::vec::fixed<12> staged;
    staged=source;
    for (std::size_t i=0;i<12;++i) destination[i]=staged[i];
}
}
#endif
