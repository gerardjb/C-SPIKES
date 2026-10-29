#ifndef CSP_HISTORY_NUMPY_BRIDGE_H
#define CSP_HISTORY_NUMPY_BRIDGE_H
#include "history_storage.h"
#include <pybind11/numpy.h>
#include <vector>

namespace csp_python {
// Outbound-only conversion under the caller's existing GIL. The transient
// descriptor never escapes: ndarray.copy() returns independently owned storage,
// preserving the previous caster's default copy order and empty-array behavior.
inline pybind11::array_t<double> numpy_copy(const csp_model::HistoryVector& source) {
    std::vector<pybind11::ssize_t> shape = {static_cast<pybind11::ssize_t>(source.size())};
    std::vector<pybind11::ssize_t> strides = {static_cast<pybind11::ssize_t>(sizeof(double))};
    pybind11::array_t<double> array(shape, strides, source.data(), pybind11::none());
    pybind11::array_t<double> copy = array.attr("copy")();
    return copy;
}
inline pybind11::array_t<double> numpy_copy(const csp_model::HistoryMatrix& source) {
    std::vector<pybind11::ssize_t> shape = {
        static_cast<pybind11::ssize_t>(source.rows()),
        static_cast<pybind11::ssize_t>(source.cols())
    };
    std::vector<pybind11::ssize_t> strides = {
        static_cast<pybind11::ssize_t>(sizeof(double)),
        static_cast<pybind11::ssize_t>(sizeof(double) * source.rows())
    };
    pybind11::array_t<double> array(shape, strides, source.data(), pybind11::none());
    pybind11::array_t<double> copy = array.attr("copy")();
    return copy;
}
}
#endif
