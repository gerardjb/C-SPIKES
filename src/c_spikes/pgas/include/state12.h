#ifndef CSP_STATE12_H
#define CSP_STATE12_H
#include <array>
#include <cstddef>

namespace csp_model {
// Owned contiguous model state. Default construction intentionally does not
// initialize coefficients; callers must fill them before reading.
struct State12 {
    std::array<double,12> coefficients;
    static constexpr std::size_t n_elem=12;
    double& operator()(std::size_t index) noexcept { return coefficients[index]; }
    const double& operator()(std::size_t index) const noexcept { return coefficients[index]; }
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
