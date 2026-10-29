#ifndef CSP_BRIGHTNESS_REDUCTION_H
#define CSP_BRIGHTNESS_REDUCTION_H
#include "bright_indices5.h"
#include <array>
#include <cstddef>
#include <stdexcept>
namespace csp_model {
// Direct indexed reductions check/read each coefficient before its addition.
inline double indexed_brightness_sum5(const double* values, std::size_t size,
                                     const BrightIndices5& indices) {
    double a = 0.0, b = 0.0;
    for (std::size_t i = 0; i < 4; i += 2) {
        if (indices[i] >= size) throw std::out_of_range("Mat::elem(): index out of bounds");
        a += values[indices[i]];
        if (indices[i+1] >= size) throw std::out_of_range("Mat::elem(): index out of bounds");
        b += values[indices[i+1]];
    }
    if (indices[4] >= size) throw std::out_of_range("Mat::elem(): index out of bounds");
    a += values[indices[4]];
    return a + b;
}
// Materialized gathers check each pair before either read, then the tail.
inline std::array<double,5> gather_brightness5(const double* values, std::size_t size,
                                             const BrightIndices5& indices) {
    std::array<double,5> result;
    for (std::size_t i = 0; i < 4; i += 2) {
        if (indices[i] >= size || indices[i+1] >= size)
            throw std::out_of_range("Mat::elem(): index out of bounds");
        result[i] = values[indices[i]];
        result[i+1] = values[indices[i+1]];
    }
    if (indices[4] >= size) throw std::out_of_range("Mat::elem(): index out of bounds");
    result[4] = values[indices[4]];
    return result;
}
inline double sum_brightness5(const std::array<double,5>& values) {
    double a = 0.0, b = 0.0;
    a += values[0]; b += values[1];
    a += values[2]; b += values[3];
    a += values[4];
    return a + b;
}
}
#endif
