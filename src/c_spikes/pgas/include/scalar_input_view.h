#ifndef CSP_SCALAR_INPUT_VIEW_H
#define CSP_SCALAR_INPUT_VIEW_H
#include <cstddef>
#include <stdexcept>
namespace csp_model {
// Borrowed only for a synchronous call. The caller owns size live contiguous
// doubles; nullptr is permitted only for size zero. No input is retained.
class ScalarInputView {
    const double* data_;
    std::size_t size_;
public:
    ScalarInputView(const double* data, std::size_t size) noexcept : data_(data), size_(size) {}
    const double* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    const double& operator()(std::size_t index) const {
        if (index >= size_) throw std::out_of_range("Mat::operator(): index out of bounds");
        return data_[index];
    }
};
}
#endif
