#ifndef CSP_APPLICATION_RESULT_COMPAT_H
#define CSP_APPLICATION_RESULT_COMPAT_H
#include "history_storage.h"
#include <array>
#include <cstring>
namespace csp_application {
// Stage the complete six-column matrix before publishing into the live member.
inline void append_parameter_row(csp_model::HistoryMatrix& destination,
                                 const std::array<double,6>& row) {
    const auto rows = destination.rows();
    csp_model::HistoryMatrix next;
    next.set_size(rows + 1, 6);
    for (std::size_t col=0; col<6; ++col) {
        if (rows) std::memcpy(next.data()+col*(rows+1),
                              destination.data()+col*rows, rows*sizeof(double));
        next(rows,col)=row[col];
    }
    destination.swap(next);
}
}
#endif
