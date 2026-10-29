#include "particle.h"
#include <type_traits>
static_assert(std::is_same_v<decltype(Particle::C),csp_application::ParticleState>);
static_assert(std::is_same_v<decltype(Particle::index),int>);
static_assert(std::is_same_v<decltype(Particle::ancestor),int>);
static_assert(std::is_same_v<decltype(Particle::S),int>);
static_assert(std::is_same_v<decltype(Particle::burst),int>);
static_assert(std::is_same_v<decltype(&SMC::PGAS),void(SMC::*)(const param&,const Trajectory&,Trajectory&)>);
static_assert(std::is_same_v<decltype(&SMC::move_and_weight),void(SMC::*)(Particle&,const Particle&,double,const param&,double,double,bool)>);
static_assert(std::is_same_v<decltype(&SMC::move_and_weight_GTS),decltype(&SMC::move_and_weight)>);

#include "GCaMP_model.h"
#include "Analyzer.h"
template<class T,class=void> struct has_alt:std::false_type{};
template<class T> struct has_alt<T,std::void_t<decltype(&T::integrateOverTime2)>>:std::true_type{};
template<class T,class=void> struct has_sim:std::false_type{};
template<class T> struct has_sim<T,std::void_t<decltype(&T::simulate)>>:std::true_type{};
static_assert(!has_alt<GCaMP>::value && !has_sim<Trajectory>::value);
static_assert(std::is_same_v<decltype(&GCaMP::integrateOverTime),void(GCaMP::*)(csp_model::ScalarInputView,csp_model::ScalarInputView)>);
static_assert(std::is_same_v<decltype(std::declval<const GCaMP&>().getPoolTotals()),std::array<double,2>>);
static_assert(std::is_same_v<decltype(std::declval<const Analyzer&>().get_run_status()),const std::string&>);
