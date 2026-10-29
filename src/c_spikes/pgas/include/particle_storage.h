#ifndef CSP_PARTICLE_STORAGE_H
#define CSP_PARTICLE_STORAGE_H
#include <vector>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
namespace csp_application {
using ParticleState = std::vector<double>;
inline double& particle_at(ParticleState& values, std::size_t index) {
    if(index >= values.size()) throw std::out_of_range("Mat::operator(): index out of bounds");
    return values[index];
}
inline const double& particle_at(const ParticleState& values, std::size_t index) {
    if(index >= values.size()) throw std::out_of_range("Mat::operator(): index out of bounds");
    return values[index];
}
inline void assign_particle_state(ParticleState& destination, const ParticleState& source) {
    if(&destination == &source) return;
    destination.resize(source.size());
    std::copy(source.begin(), source.end(), destination.begin());
}
}
#endif
