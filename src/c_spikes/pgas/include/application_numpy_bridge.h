#ifndef CSP_APPLICATION_NUMPY_BRIDGE_H
#define CSP_APPLICATION_NUMPY_BRIDGE_H
#include "history_numpy_bridge.h"
namespace pybind11 { namespace detail {
// Outbound-only live native result conversion: Python still owns a fresh copy.
template <> struct type_caster<csp_model::HistoryMatrix> {
    PYBIND11_TYPE_CASTER(csp_model::HistoryMatrix, _("numpy.ndarray"));
    bool load(handle, bool) { return false; }
    static handle cast(const csp_model::HistoryMatrix& source,
                       return_value_policy, handle) {
        return csp_python::numpy_copy(source).release();
    }
};
}}
#endif
