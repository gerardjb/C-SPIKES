#ifndef CSP_APPLICATION_ARMA_BRIDGE_H
#define CSP_APPLICATION_ARMA_BRIDGE_H
#include "application_observation.h"
#include <armadillo>
namespace csp_application {
// Owned, call-local snapshot at the existing residual expression only.
inline arma::vec owned_arma(const ObservationVector& source) {
    return arma::vec(source);
}
}
#endif
