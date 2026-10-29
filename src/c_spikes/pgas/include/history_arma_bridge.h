#ifndef CSP_HISTORY_ARMA_BRIDGE_H
#define CSP_HISTORY_ARMA_BRIDGE_H
#include "history_storage.h"
#include <armadillo>
#include <algorithm>
#include <limits>

namespace csp_model {
static_assert(std::numeric_limits<arma::uword>::digits == std::numeric_limits<std::size_t>::digits,
              "History bridge requires the qualified native index width");
inline arma::mat owned_arma(const HistoryMatrix& m) {
    arma::mat result(m.rows(),m.cols());
    if (m.size()) std::copy(m.data(),m.data()+m.size(),result.begin());
    return result;
}
inline arma::vec owned_column(const HistoryMatrix& m,std::size_t col) {
    arma::vec result(m.rows());
    for (std::size_t row=0;row<m.rows();++row) result(row)=m(row,col);
    return result;
}
inline void copy_row(HistoryMatrix& m,std::size_t row,const arma::vec& values) {
    for (std::size_t col=0;col<m.cols();++col) m(row,col)=values(col);
}
inline void copy_column(HistoryMatrix& m,std::size_t col,const arma::vec& values) {
    for (std::size_t row=0;row<m.rows();++row) m(row,col)=values(row);
}
inline arma::vec owned_arma(const HistoryVector& v) {
    arma::vec result(v.size());
    if (v.size()) std::copy(v.data(),v.data()+v.size(),result.begin());
    return result;
}
inline void copy_from_arma(HistoryVector& v,const arma::vec& values) {
    v.set_size(values.n_elem);
    for (std::size_t i=0;i<v.size();++i) v(i)=values(i);
}
}
#endif
