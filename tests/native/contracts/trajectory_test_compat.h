#pragma once
#include <armadillo>
#include "particle.h"
#include <algorithm>
template<class T> auto& value_at(T& v,size_t i){return csp_application::trajectory_at(v,i);}
template<class T> const auto& value_at(const T& v,size_t i){return csp_application::trajectory_at(v,i);}
template<class T> auto pointer(const T& v){return v.data();}
template<class T> size_t count_values(const T& v){return v.size();}
inline auto pointer(const arma::vec& v){return v.memptr();}
inline size_t count_values(const arma::vec& v){return v.n_elem;}
template<class T,class U> void fill_values(T& v,U x){std::fill(v.begin(),v.end(),x);}
inline double& particle_value(arma::vec& v,size_t i){return v(i);}
inline const double& particle_value(const arma::vec& v,size_t i){return v(i);}
inline double& particle_value(csp_application::ParticleState& v,size_t i){return csp_application::particle_at(v,i);}
inline const double& particle_value(const csp_application::ParticleState& v,size_t i){return csp_application::particle_at(v,i);}
