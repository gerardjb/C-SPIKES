#ifndef CSP_SCALAR_INPUT_NUMPY_BRIDGE_H
#define CSP_SCALAR_INPUT_NUMPY_BRIDGE_H
#include "scalar_input_view.h"
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>
#include <iostream>
namespace csp_python {
// Python inputs retain the former logical-element copy, including negative,
// zero and unaligned strides. The model borrows this owner only during the call.
struct ScalarInputCopy {
    std::vector<double> values;
    csp_model::ScalarInputView view() const noexcept { return {values.data(), values.size()}; }
};
}
namespace pybind11 { namespace detail {
template <> struct type_caster<csp_python::ScalarInputCopy> {
    PYBIND11_TYPE_CASTER(csp_python::ScalarInputCopy, _("numpy.ndarray"));
    bool load(handle src, bool) {
        if (!isinstance<array_t<double>>(src)) {
            std::cout << "Not a numpy array of doubles." << std::endl;
            return false;
        }
        array_t<double> buf = array_t<double>::ensure(src);
        if (!buf) {
            std::cout << "Failed to ensure numpy array." << std::endl;
            return false;
        }
        if (buf.ndim() != 1) {
            std::cout << "Array is not 1-dimensional." << std::endl;
            return false;
        }
        const std::size_t size = buf.shape(0);
        value.values.resize(size);
        if (size != 0) {
            const auto* data = reinterpret_cast<const char*>(buf.data());
            const ssize_t stride = buf.strides(0);
            if (stride == static_cast<ssize_t>(sizeof(double))) {
                std::memcpy(value.values.data(), data, size * sizeof(double));
            } else {
                for (ssize_t i = 0; i < buf.shape(0); ++i)
                    std::memcpy(value.values.data() + i, data + i * stride, sizeof(double));
            }
        }
        return true;
    }
};
}}
#endif
