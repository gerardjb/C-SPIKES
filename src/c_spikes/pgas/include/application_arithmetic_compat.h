#ifndef CSP_APPLICATION_ARITHMETIC_COMPAT_H
#define CSP_APPLICATION_ARITHMETIC_COMPAT_H
#include "application_observation.h"
#include "trajectory_storage.h"
#include <cmath>
#include <limits>
#include <string>
namespace csp_application {
using SamplingIndices = std::vector<std::uint64_t>;
inline SamplingIndices find_burst_indices(const TrajectoryBursts& values, std::int64_t value) {
    SamplingIndices indices(values.size());
    std::size_t count=0;
    for(std::size_t i=0;i<values.size();++i) if(values[i]==value) indices[count++]=i;
    indices.resize(count);
    return indices;
}
inline double gathered_sum_compat(const TrajectorySamples& values, const SamplingIndices& indices) {
    auto at=[&](std::size_t i) {
        const auto index=indices[i];
        if(index>=values.size()) throw std::out_of_range("Mat::elem(): index out of bounds");
        return values[index];
    };
    double acc1=0.,acc2=0.;std::size_t i=0,j=1;
    for(;j<indices.size();i+=2,j+=2){acc1+=at(i);acc2+=at(j);}
    if(i<indices.size()) acc1+=at(i);
    return acc1+acc2;
}
inline double sample_sum_compat(const TrajectorySamples& values) {
    double acc1=0.,acc2=0.;std::size_t i=0,j=1;
    for(;j<values.size();i+=2,j+=2){acc1+=values[i];acc2+=values[j];}
    if(i<values.size()) acc1+=values[i];
    return acc1+acc2;
}
inline double residual_norm_compat(const ObservationVector& observations,
                                   const TrajectorySamples& baseline,
                                   const TrajectorySamples& fluorescence) {
    const auto y=observations,b=baseline,c=fluorescence;
    auto shape=[](std::size_t a,std::size_t z) {
        if(a!=z) throw std::logic_error("subtraction: incompatible matrix dimensions: "+std::to_string(a)+"x1 and "+std::to_string(z)+"x1");
    };
    shape(y.size(),b.size());shape(y.size(),c.size());
    const auto n=y.size();if(n==0)return 0.;
    auto residual=[&](std::size_t i){return (y[i]-b[i])-c[i];};
    double acc1=0.,acc2=0.;std::size_t i=0,j=1;
    for(;j<n;i+=2,j+=2){const double a=residual(i),z=residual(j);acc1+=a*a;acc2+=z*z;}
    if(i<n){const double a=residual(i);acc1+=a*a;}
    const double root=std::sqrt(acc1+acc2);
    if(root!=0. && std::isfinite(root))return root;
    TrajectorySamples values(n);for(i=0;i<n;++i)values[i]=residual(i);
    double maximum=-std::numeric_limits<double>::infinity();
    for(i=0;i<n;++i){const double a=std::abs(values[i]);if(a>maximum)maximum=a;}
    if(maximum==0.)return 0.;
    acc1=0.;acc2=0.;i=0;j=1;
    for(;j<n;i+=2,j+=2){double a=values[i],z=values[j];a/=maximum;z/=maximum;acc1+=a*a;acc2+=z*z;}
    if(i<n){const double a=values[i]/maximum;acc1+=a*a;}
    const double result=std::sqrt(acc1+acc2)*maximum;
    return result<=0. ? 0. : result;
}
}
#endif
