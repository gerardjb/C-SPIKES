#ifndef CSP_TERMINAL_WEIGHT_COPY_H
#define CSP_TERMINAL_WEIGHT_COPY_H

#include <Kokkos_Core.hpp>

namespace csp_pgas_detail {

// A named callable avoids CUDA extended-lambda restrictions in private methods.
// View handles are copied, not their allocations; dispatch and fencing stay with
// the caller.
template<class HistoryView, class TerminalView>
struct TerminalWeightCopy {
    HistoryView history;
    TerminalView terminal;
    unsigned int last;

    KOKKOS_INLINE_FUNCTION
    void operator()(int j) const {
        terminal(j) = history(j, last);
    }
};

} // namespace csp_pgas_detail

#endif
