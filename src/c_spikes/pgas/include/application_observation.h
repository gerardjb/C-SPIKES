#ifndef CSP_APPLICATION_OBSERVATION_H
#define CSP_APPLICATION_OBSERVATION_H
#include <vector>
#include <stdexcept>
#include <cstddef>
namespace csp_application {
using ObservationVector = std::vector<double>;
inline double& observation_at(ObservationVector& values, std::size_t index) {
    if (index >= values.size()) throw std::out_of_range("Mat::operator(): index out of bounds");
    return values[index];
}
inline const double& observation_at(const ObservationVector& values, std::size_t index) {
    if (index >= values.size()) throw std::out_of_range("Mat::operator(): index out of bounds");
    return values[index];
}
}
#endif
