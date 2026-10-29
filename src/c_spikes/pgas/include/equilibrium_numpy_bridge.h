#ifndef CSP_EQUILIBRIUM_NUMPY_BRIDGE_H
#define CSP_EQUILIBRIUM_NUMPY_BRIDGE_H
#include <array>
#include <vector>
#include <pybind11/numpy.h>

namespace csp_python {
// Binding-private owned result. Only completed equilibrium results populate it.
struct EquilibriumOutput12 {
    std::array<double,12> values;
};
}
namespace pybind11 { namespace detail {
template <> struct type_caster<csp_python::EquilibriumOutput12> {
    PYBIND11_TYPE_CASTER(csp_python::EquilibriumOutput12, _("numpy.ndarray"));
    bool load(handle, bool) { return false; }
    static handle cast(const csp_python::EquilibriumOutput12& source,
                       return_value_policy, handle) {
        // Keep allocation at the existing cast site, after the legacy call guard.
        // The transient descriptor cannot escape; Python receives its owned copy.
        std::vector<pybind11::ssize_t> shape = {12};
        std::vector<pybind11::ssize_t> strides = {
            static_cast<pybind11::ssize_t>(sizeof(double))
        };
        pybind11::array_t<double> array(shape, strides, source.values.data(), pybind11::none());
        pybind11::array_t<double> copy = array.attr("copy")();
        return copy.release();
    }
};
}}
#endif
