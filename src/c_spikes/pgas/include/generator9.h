#ifndef CSP_GENERATOR9_H
#define CSP_GENERATOR9_H
#include <array>
#include <cstddef>

namespace csp_model {
// Owned column-major generator. Default construction intentionally does not
// initialize coefficients; callers must fill them before reading.
struct Generator9 {
    std::array<double,81> coefficients;
    static constexpr std::size_t n_elem=81;
    double& operator()(std::size_t row,std::size_t col) noexcept { return coefficients[row+9*col]; }
    const double& operator()(std::size_t row,std::size_t col) const noexcept { return coefficients[row+9*col]; }
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
}
#endif
