#ifndef CSP_PARAMETER_BLOCK_H
#define CSP_PARAMETER_BLOCK_H
#include <array>
#include <cstddef>
#include <stdexcept>

namespace csp_model {
// Owned parameter coefficients. Zero initialization preserves the pinned
// Armadillo fixed-vector constructor; raw records need not track model scalars.
template<std::size_t N> struct ParameterBlock {
    std::array<double,N> coefficients{};
    static constexpr std::size_t n_elem=N;
    double& operator()(std::size_t index) {
        if(index>=N) throw std::out_of_range("Col::operator(): index out of bounds");
        return coefficients[index];
    }
    const double& operator()(std::size_t index) const {
        if(index>=N) throw std::out_of_range("Col::operator(): index out of bounds");
        return coefficients[index];
    }
    double& operator[](std::size_t index) noexcept { return coefficients[index]; }
    const double& operator[](std::size_t index) const noexcept { return coefficients[index]; }
    double* data() noexcept { return coefficients.data(); }
    const double* data() const noexcept { return coefficients.data(); }
    auto begin() noexcept { return coefficients.begin(); }
    auto end() noexcept { return coefficients.end(); }
    auto begin() const noexcept { return coefficients.begin(); }
    auto end() const noexcept { return coefficients.end(); }
    void fill(double value) noexcept { coefficients.fill(value); }
};
using IndicatorParameters14=ParameterBlock<14>;
using CellParameterRecord6=ParameterBlock<6>;
}
#endif
