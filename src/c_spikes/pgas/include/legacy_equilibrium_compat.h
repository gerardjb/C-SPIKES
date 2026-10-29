#ifndef CSP_LEGACY_EQUILIBRIUM_COMPAT_H
#define CSP_LEGACY_EQUILIBRIUM_COMPAT_H
#include "generator9.h"
#include "state12.h"
namespace csp_model {
// Private compatibility boundary: retains the pinned legacy SVD and policy.
State12 legacy_equilibrium12_compat(const Generator9& generator,
    double indicator_total, double stored_buffer_bound, double calcium);
}
#endif
