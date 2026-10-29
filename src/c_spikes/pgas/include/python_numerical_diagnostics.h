#ifndef CSP_PYTHON_NUMERICAL_DIAGNOSTICS_H
#define CSP_PYTHON_NUMERICAL_DIAGNOSTICS_H
#include <pybind11/pybind11.h>
#include "numerical_diagnostics.h"

namespace csp_diagnostics {
#if C_SPIKES_OBSERVATIONAL_DIAGNOSTICS
// Installed only for the duration of a bound call, never as a process-global sink.
// All logging/formatting occurs at the outer host boundary, with the GIL held.
inline void python_sink(const Record& record) noexcept {
    try {
        pybind11::gil_scoped_acquire gil;
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        try {
            char first[1024], summary[768], debug[512];
            format(record, first, sizeof(first), summary, sizeof(summary), debug, sizeof(debug));
            auto logger = pybind11::module_::import("logging").attr("getLogger")("c_spikes.pgas");
            logger.attr("warning")("%s", first);
            logger.attr("warning")("%s", summary);
            if (logger.attr("isEnabledFor")(10).cast<bool>()) logger.attr("debug")("%s", debug);
        } catch (pybind11::error_already_set& error) {
            // A broken user handler must not alter the numerical call or replace
            // an exception already being unwound. Do not print a fallback duplicate.
            error.restore();
            PyErr_Clear();
        } catch (...) {
            PyErr_Clear();
        }
        PyErr_Restore(type, value, traceback);
    } catch (...) {}
}
class PythonScope {
    Scope scope_;
public:
    PythonScope() noexcept : scope_(Scope::python_operation(), python_sink) {}
};
#else
class PythonScope {};
#endif
} // namespace csp_diagnostics
#endif
