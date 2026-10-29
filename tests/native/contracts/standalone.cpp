// Original tests/native/g2e/standalone.cpp
#include "generator9.h"
#include <type_traits>
static_assert(sizeof(csp_model::Generator9)==81*sizeof(double));
static_assert(std::is_trivially_copyable<csp_model::Generator9>::value);
static_assert(std::is_standard_layout<csp_model::Generator9>::value);

// Original tests/native/g2g/standalone.cpp
#include "state12.h"
#include <type_traits>
static_assert(sizeof(csp_model::State12)==12*sizeof(double));
static_assert(std::is_trivially_copyable_v<csp_model::State12>);

// Original tests/native/g2i/standalone.cpp
#include "parameter_block.h"
#include <type_traits>
static_assert(sizeof(csp_model::IndicatorParameters14)==14*sizeof(double));
static_assert(sizeof(csp_model::CellParameterRecord6)==6*sizeof(double));
static_assert(std::is_standard_layout<csp_model::IndicatorParameters14>::value);

// Original tests/native/g2j/standalone.cpp
#include "bright_indices5.h"
#include <tuple>
static_assert(std::tuple_size<csp_model::BrightIndices5>::value==5);

// Original tests/native/g2l/standalone.cpp
#include "history_storage.h"
#include <type_traits>
static_assert(std::is_copy_constructible<csp_model::HistoryMatrix>::value,"owned copy");
static_assert(std::is_nothrow_move_constructible<csp_model::HistoryMatrix>::value,"move");

// Original tests/native/g2m/standalone.cpp
#include "history_storage.h"
#include <type_traits>
static_assert(std::is_copy_constructible<csp_model::HistoryVector>::value,"copy");
static_assert(std::is_nothrow_move_constructible<csp_model::HistoryVector>::value,"move");

// Original tests/native/g2o/standalone.cpp
#include "history_numpy_bridge.h"
#include <type_traits>
static_assert(std::is_same<decltype(csp_python::numpy_copy(std::declval<const csp_model::HistoryVector&>())),pybind11::array_t<double>>::value,"vector output");
static_assert(std::is_same<decltype(csp_python::numpy_copy(std::declval<const csp_model::HistoryMatrix&>())),pybind11::array_t<double>>::value,"matrix output");
pybind11::array_t<double> proof(const csp_model::HistoryVector& v){return csp_python::numpy_copy(v);}

// Original tests/native/g2r/standalone.cpp
#include "scalar_input_view.h"
#include <type_traits>
static_assert(std::is_trivially_copyable<csp_model::ScalarInputView>::value,"borrowed value");
static_assert(sizeof(csp_model::ScalarInputView)==sizeof(const double*)+sizeof(std::size_t),"pointer/size only");
static_assert(std::is_same<decltype(std::declval<csp_model::ScalarInputView>()(0)),const double&>::value,"read only");
const double* proof(const double* p,std::size_t n){return csp_model::ScalarInputView(p,n).data();}

// Original tests/native/g2t/standalone.cpp
#include "scalar_output_view.h"
#include <type_traits>
static_assert(std::is_trivially_copyable<csp_model::ScalarOutputView>::value,"borrowed value");
static_assert(sizeof(csp_model::ScalarOutputView)==sizeof(double*)+sizeof(std::size_t),"pointer/size only");
static_assert(std::is_same<decltype(std::declval<csp_model::ScalarOutputView>()(0)),double&>::value,"writable");
double* proof(double* p,std::size_t n){return csp_model::ScalarOutputView(p,n).data();}

// Original tests/native/g2ab/standalone.cpp
#include "equilibrium_numpy_bridge.h"
#include <type_traits>
static_assert(sizeof(csp_python::EquilibriumOutput12)==12*sizeof(double),"owned fixed record");
static_assert(std::is_same<decltype(csp_python::EquilibriumOutput12{}.values),std::array<double,12>>::value,"owned array");
void header_contract(){}
#include "particle.h"
#include "Analyzer.h"
#include <fstream>
#include <utility>
using Observation=csp_application::ObservationVector;
static_assert(std::is_constructible_v<SMC,Observation,Observation,int,constpar&,bool,int,unsigned int,std::string>);
static_assert(!std::is_constructible_v<SMC,Observation&,constpar&,bool>);
static_assert(!std::is_constructible_v<SMC,std::string,int,constpar&,bool,int,unsigned int,std::string>);
static_assert(std::is_constructible_v<Trajectory,unsigned int,std::string>);
static_assert(std::is_same_v<decltype(&Trajectory::write),void(Trajectory::*)(std::ofstream&,unsigned int)>);
static_assert(std::is_same_v<decltype(&Trajectory::logp),double(Trajectory::*)(const param*,const constpar*)>);
static_assert(std::is_same_v<decltype(std::declval<Trajectory&>()=std::declval<const Trajectory&>()),Trajectory&>);
static_assert(std::is_same_v<decltype(Trajectory::B),csp_application::TrajectorySamples>);
static_assert(std::is_same_v<decltype(Trajectory::C),csp_application::TrajectorySamples>);
static_assert(std::is_same_v<decltype(Trajectory::S),csp_application::TrajectorySamples>);
static_assert(std::is_same_v<decltype(Trajectory::Y),csp_application::TrajectorySamples>);
static_assert(std::is_same_v<decltype(Trajectory::burst),csp_application::TrajectoryBursts>);
static_assert(sizeof(csp_application::TrajectoryBursts::value_type)==8);
static_assert(std::is_same_v<decltype(std::declval<const Analyzer&>().get_parameter_samples()),const csp_model::HistoryMatrix&>);
