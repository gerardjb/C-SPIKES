#ifndef CSP_SCALAR_OUTPUT_VIEW_H
#define CSP_SCALAR_OUTPUT_VIEW_H
#include <cstddef>
#include <stdexcept>
namespace csp_model {
// Borrowed live contiguous storage for a synchronous call. Never resized or
// retained. A failing later write leaves earlier writes intact, as before.
class ScalarOutputView {
    double* data_;
    std::size_t size_;
public:
    ScalarOutputView(double* data, std::size_t size) noexcept : data_(data), size_(size) {}
    double* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    double& operator()(std::size_t index) const {
        if (index >= size_) throw std::out_of_range("Mat::operator(): index out of bounds");
        return data_[index];
    }
};
}
#endif
