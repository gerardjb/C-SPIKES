#ifndef CSP_MODEL_WORKSPACE_H
#define CSP_MODEL_WORKSPACE_H
#include "state12.h"
#include "history_storage.h"
#include "parameter_block.h"
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace csp_model { namespace workspace {
// Private call-local workspaces; no native field or public ABI is changed.
struct Indicator9 {
    std::array<double,9> values;
    static constexpr std::size_t n_elem=9;
    double* data() noexcept { return values.data(); }
    const double* data() const noexcept { return values.data(); }
    double* memptr() noexcept { return data(); }
    const double* memptr() const noexcept { return data(); }
    double& operator[](std::size_t i) noexcept { return values[i]; }
    const double& operator[](std::size_t i) const noexcept { return values[i]; }
    double& operator()(std::size_t i) { if(i>=9)throw std::out_of_range("Mat::operator(): index out of bounds");return values[i]; }
    const double& operator()(std::size_t i) const { if(i>=9)throw std::out_of_range("Mat::operator(): index out of bounds");return values[i]; }
};
// Adapted from Armadillo eglue_core_meat.hpp (Apache-2.0); preserve paired
// loads, multiplication operand order, stores and the odd tail.
inline void advance9_compat(Indicator9& state,double dt,const Indicator9& derivative) {
    std::size_t i=0,j=1;
    for(;j<9;i+=2,j+=2){
        double a=state[i],b=state[j];
        a+=derivative[i]*dt;b+=derivative[j]*dt;
        state[i]=a;state[j]=b;
    }
    if(i<9)state[i]=state[i]+derivative[i]*dt;
}
inline Indicator9 indicator_copy(const State12& state) { Indicator9 out;for(std::size_t i=0;i<9;++i)out[i]=state[i];return out; }
inline HistoryVector positive_grid_compat(double start,double step,double end) {
    HistoryVector out;
    if(start>end)return out;
    const double M=(end-start)/step;
    const std::size_t count=std::size_t(1)+std::size_t(std::floor(double(M)));
    out.set_size(count);
    for(std::size_t i=0;i<count;++i)out(i)=double(start+double(i*step));
    return out;
}
inline std::size_t fill_spike_matches(HistoryVector& input,const HistoryVector& grid,double spike,double half_step,double amplitude) {
    std::size_t count=0;
    for(std::size_t i=0;i<grid.size();++i)if(std::abs(grid(i)-spike)<=half_step){input(i)=amplitude;++count;}
    return count;
}
class NumericInputSnapshot {
    std::vector<double> values_;
public:
    NumericInputSnapshot(const double* data,std::size_t n):values_(n){for(std::size_t i=0;i<n;++i)values_[i]=data[i];}
    std::size_t size() const noexcept {return values_.size();}
    double operator()(std::size_t i) const {if(i>=size())throw std::out_of_range("Mat::operator(): index out of bounds");return values_[i];}
};
inline void assign_indicator14(ParameterBlock<14>& out,const NumericInputSnapshot& in) {
    if(in.size()!=14)throw std::logic_error("Mat::init(): size is fixed and hence cannot be changed");
    for(std::size_t i=0;i<14;++i)out[i]=in(i);
}
}}
#endif
