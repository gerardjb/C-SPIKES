#ifndef CSP_TRAJECTORY_STORAGE_H
#define CSP_TRAJECTORY_STORAGE_H
#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
namespace csp_application {
using TrajectorySamples = std::vector<double>;
using TrajectoryBursts = std::vector<std::int64_t>;
template<class T> inline T& trajectory_at(std::vector<T>& values, std::size_t index) {
    if (index >= values.size()) throw std::out_of_range("Mat::operator(): index out of bounds");
    return values[index];
}
template<class T> inline const T& trajectory_at(const std::vector<T>& values, std::size_t index) {
    if (index >= values.size()) throw std::out_of_range("Mat::operator(): index out of bounds");
    return values[index];
}
// Preserve same-size storage and sequential publication, without a transaction.
// Native references must be reacquired after a size-changing replacement.
template<class T> inline void assign_trajectory_values(std::vector<T>& destination, const std::vector<T>& source) {
    if (&destination == &source) return;
    destination.resize(source.size());
    std::copy(source.begin(), source.end(), destination.begin());
}
}
#endif
