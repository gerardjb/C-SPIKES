#ifndef CSP_HISTORY_STORAGE_H
#define CSP_HISTORY_STORAGE_H
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace csp_model {
// Owned column-major history storage. New elements are intentionally uninitialized;
// producers must write them before numerical observation, as with the old set_size.
class HistoryMatrix {
    std::size_t rows_ = 0, cols_ = 0;
    std::unique_ptr<double[]> values_;
public:
    HistoryMatrix() noexcept = default;
    HistoryMatrix(const HistoryMatrix& other) {
        set_size(other.rows_, other.cols_);
        if (size()) std::memcpy(data(), other.data(), size()*sizeof(double));
    }
    HistoryMatrix& operator=(const HistoryMatrix& other) {
        if (this != &other) { HistoryMatrix copy(other); swap(copy); }
        return *this;
    }
    HistoryMatrix(HistoryMatrix&& other) noexcept { swap(other); }
    HistoryMatrix& operator=(HistoryMatrix&& other) noexcept {
        if (this != &other) { reset(); swap(other); }
        return *this;
    }
    void swap(HistoryMatrix& other) noexcept {
        std::swap(rows_, other.rows_); std::swap(cols_, other.cols_);
        values_.swap(other.values_);
    }
    void reset() noexcept { values_.reset(); rows_=cols_=0; }
    void set_size(std::size_t rows, std::size_t cols) {
        constexpr auto max_count = std::numeric_limits<std::size_t>::max()/sizeof(double);
        if (cols && rows > max_count/cols) throw std::bad_array_new_length();
        const auto count = rows*cols;
        if (count != size()) {
            std::unique_ptr<double[]> next(count ? new double[count] : nullptr);
            values_.swap(next);
        }
        rows_=rows; cols_=cols;
    }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return rows_*cols_; }
    double* data() noexcept { return values_.get(); }
    const double* data() const noexcept { return values_.get(); }
    double& operator()(std::size_t row,std::size_t col) noexcept { return values_[row+rows_*col]; }
    const double& operator()(std::size_t row,std::size_t col) const noexcept { return values_[row+rows_*col]; }
};
// A one-dimensional owning history. Reuse the independently owned matrix
// allocation/copy implementation; no Armadillo storage or cached view is held.
class HistoryVector {
    HistoryMatrix values_;
public:
    HistoryVector() noexcept = default;
    HistoryVector(const HistoryVector&) = default;
    HistoryVector& operator=(const HistoryVector&) = default;
    HistoryVector(HistoryVector&&) noexcept = default;
    HistoryVector& operator=(HistoryVector&&) noexcept = default;
    void reset() noexcept { values_.reset(); }
    void set_size(std::size_t size) { values_.set_size(size, 1); }
    std::size_t size() const noexcept { return values_.size(); }
    double* data() noexcept { return values_.data(); }
    const double* data() const noexcept { return values_.data(); }
    double& operator()(std::size_t index) noexcept { return values_(index, 0); }
    const double& operator()(std::size_t index) const noexcept { return values_(index, 0); }
};
// A non-owning scan descriptor with the existing diagnostics protocol. Its lifetime
// is limited to the immediate scan; it is not an exposed or cached history view.
struct HistoryScan {
    const double* pointer;
    std::size_t n_elem;
    const double& operator[](std::size_t index) const noexcept { return pointer[index]; }
};
inline HistoryScan scan_view(const HistoryMatrix& m) noexcept { return {m.data(),m.size()}; }
inline HistoryScan scan_view(const HistoryVector& v) noexcept { return {v.data(),v.size()}; }
}
#endif
