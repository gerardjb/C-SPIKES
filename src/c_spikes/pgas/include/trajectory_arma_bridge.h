#ifndef CSP_TRAJECTORY_ARMA_BRIDGE_H
#define CSP_TRAJECTORY_ARMA_BRIDGE_H
#include "trajectory_storage.h"
#include <armadillo>
#include <limits>
namespace csp_application {
// Private call-local owning snapshots; never retained or written back implicitly.
inline arma::vec owned_samples(const TrajectorySamples& source) { return arma::vec(source); }
inline arma::Col<arma::sword> owned_bursts(const TrajectoryBursts& source) {
    static_assert(std::numeric_limits<arma::sword>::is_signed &&
                  std::numeric_limits<arma::sword>::digits == std::numeric_limits<std::int64_t>::digits,
                  "Trajectory requires the pinned signed 64-bit Armadillo integer range");
    arma::Col<arma::sword> result(source.size());
    for (std::size_t i=0; i<source.size(); ++i) result(i)=source[i];
    return result;
}
}
#endif
