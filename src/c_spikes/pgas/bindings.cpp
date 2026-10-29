//Required pybind libraries for expected conversions
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//External libraries called by pgas
#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <json/json.h>
#include <fstream>
#include <iostream>
//Type conversion for armadillo library in pybind11
#include "include/arma_pybind11.h"
//Bound modules
#include "include/Analyzer.h"
//In case any other functions or classes need to be exposed to python
#include "include/GCaMP_model.h"
#include "include/python_numerical_diagnostics.h"
#ifdef C_SPIKES_CHECKED_TESTING
#include "test_access.h"
#endif
// Attempting kokkos integration
#include <Kokkos_Core.hpp>
namespace py = pybind11;

#ifndef PGAS_PYBIND_MODULE
#define PGAS_PYBIND_MODULE pgas_bound
#endif

/* //method to extract final array entries as numpy array
py::array_t<double> get_final_params(Analyzer& analyzer) {
  // Create a NumPy array from the std::vector
  py::array_t<double> result(analyzer.final_params.size());
  std::copy(analyzer.final_params.begin(), analyzer.final_params.end(), result.mutable_data());
  return result;
} */

auto cleanup_callback = []() {
	// perform cleanup here -- this function is called with the GIL held
	// You must call finalize() after you are done using Kokkos.
	if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
		Kokkos::finalize();
	}
};

PYBIND11_MODULE(PGAS_PYBIND_MODULE, m) {
  // Initialize kokkos on binding to prevent finalize() call prior to initialization
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized())
        Kokkos::initialize();

	// bindings for Analyzer.cpp
		py::class_<Analyzer>(m, "Analyzer")
        .def(py::init<const arma::vec&, const arma::vec&, const std::string&, const std::string&, unsigned int, const std::string&, unsigned int,
                      const std::string&, bool, unsigned int, bool, const arma::vec&, bool, bool, unsigned int, const std::string&, int>(),
             py::arg("time"), py::arg("data"), py::arg("constants_file"), py::arg("output_folder"), py::arg("column"), py::arg("tag"),
             py::arg("niter") = 0, py::arg("trainedPriorFile") = "", py::arg("append") = false, py::arg("trim") = 1,
             py::arg("verbose") = true, py::arg("gtSpikes") = 0, py::arg("has_trained_priors") = false, py::arg("has_gtspikes") = false,
             py::arg("maxlen") = 0, py::arg("Gparam_file") = "", py::arg("seed") = 0)
        .def("run", &Analyzer::run, py::call_guard<csp_diagnostics::PythonScope>())
        .def("add_parameter_sample", &Analyzer::add_parameter_sample)
        .def("get_parameter_samples", &Analyzer::get_parameter_samples);
		    //.def("get_final_params", &get_final_params, "Get final parameters as a NumPy array");
	
				
	// bindings for GCaMP_model.cpp
    py::enum_<TimeStepMode>(m, "TimeStepMode")
        .value("FIXED", FIXED)
        .value("FIXEDLA", FIXEDLA);
		auto gcamp = py::class_<GCaMP>(m, "GCaMP")
        .def(py::init<double, double, double, double, double, double, std::string>(), py::call_guard<csp_diagnostics::PythonScope>())
				.def(py::init<std::string, std::string>(), py::arg("Gparam_file"), py::arg("Cparam_file"), py::call_guard<csp_diagnostics::PythonScope>())
				.def(py::init<const arma::vec,const arma::vec>(), py::arg("Gparams_in"), py::arg("Cparams_in"), py::call_guard<csp_diagnostics::PythonScope>())
        .def("setParams", &GCaMP::setParams, py::call_guard<csp_diagnostics::PythonScope>())
        .def("setGmat", &GCaMP::setGmat, py::call_guard<csp_diagnostics::PythonScope>())
        .def("flux", &GCaMP::flux, py::call_guard<csp_diagnostics::PythonScope>())
        .def("steady_state", &GCaMP::steady_state, py::call_guard<csp_diagnostics::PythonScope>())
        .def("init", &GCaMP::init, py::call_guard<csp_diagnostics::PythonScope>())
        .def("setState", &GCaMP::setState, py::call_guard<csp_diagnostics::PythonScope>())
        .def("setTimeStepMode", &GCaMP::setTimeStepMode, py::arg("mode").noconvert(), py::call_guard<csp_diagnostics::PythonScope>())
        .def("evolve", py::overload_cast<double, int, const arma::vec&>(&GCaMP::evolve), py::call_guard<csp_diagnostics::PythonScope>())
        .def("evolve", py::overload_cast<double, int>(&GCaMP::evolve), py::call_guard<csp_diagnostics::PythonScope>())
        .def("fixedStep", &GCaMP::fixedStep, py::call_guard<csp_diagnostics::PythonScope>())
        .def("fixedStep_LA", &GCaMP::fixedStep_LA, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getDFF", py::overload_cast<>(&GCaMP::getDFF), py::call_guard<csp_diagnostics::PythonScope>())
        .def("getDFF", py::overload_cast<const arma::vec&>(&GCaMP::getDFF), py::call_guard<csp_diagnostics::PythonScope>())
        .def("getAmplitude", &GCaMP::getAmplitude, py::call_guard<csp_diagnostics::PythonScope>())
        .def("integrateOverTime", &GCaMP::integrateOverTime, py::arg("time"), py::arg("spike_times"), py::call_guard<csp_diagnostics::PythonScope>())
        .def("integrateOverTime2", &GCaMP::integrateOverTime2, py::arg("time"), py::arg("spike_times"), py::call_guard<csp_diagnostics::PythonScope>())
        .def("getDFFValues", &GCaMP::getDFFValues, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getGValues", &GCaMP::getGValues, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getStates", [](const GCaMP &g) {
            csp_diagnostics::Scope numerical_states("GCaMP.getStates");
            py::dict states;
            states["G"] = g.getGValues();
            states["BCa"] = g.getBCaValues();
            states["Ca"] = g.getCaValues();
            states["Ca_in"] = g.getCaInValues();
            return states;
        }, py::call_guard<csp_diagnostics::PythonScope>());
#ifdef C_SPIKES_CHECKED_CPU
    using checked::Status;
    py::enum_<Status>(m,"CheckedStatus")
#define STATUS(x) .value(#x,Status::x)
      STATUS(success) STATUS(input_invalid) STATUS(backend_failed) STATUS(malformed_factors)
      STATUS(svd_quality_failed) STATUS(no_nullspace) STATUS(nonunique) STATUS(rank_ambiguous)
      STATUS(normalization_unsafe) STATUS(analytic_no_selection) STATUS(invariant_failed)
      STATUS(context_required) STATUS(context_busy);
#undef STATUS
    py::class_<checked::Context>(m,"CheckedContext")
      .def(py::init<>()).def_static("owned_isolated_process",&checked::Context::owned_isolated_process);
    py::class_<checked::Report>(m,"CheckedReport")
      .def_readonly("status",&checked::Report::status).def_readonly("committed",&checked::Report::committed)
      .def_readonly("solve_count",&checked::Report::solve_count).def_readonly("backend_code",&checked::Report::backend_code)
      .def_property_readonly("diagnostics",[](const checked::Report& r){
       py::dict d;d["operation"]=int(r.operation);d["callback_count"]=r.callback.count;d["reason"]=r.callback.reason.data();d["file"]=r.callback.file.data();d["line"]=r.callback.line;d["callback_code"]=r.callback.code;d["truncated"]=r.callback.truncated;
       py::list solves;for(int i=0;i<r.solve_count&&i<4;++i){const auto& x=r.solves[i];py::dict z;z["rank"]=x.ranks;z["singular_values"]=x.factors.s;z["residual"]=x.residual;z["mass_error"]=x.mass_error;z["minimum"]=x.minimum;z["normalization_sum"]=x.sum;z["normalization_guard"]=x.guard;z["has_state"]=x.has_state;solves.append(z);}d["solves"]=solves;return d;
      });
    m.def("try_create_gcamp",[](const arma::vec& raw,const arma::vec& cp,const checked::Context& c){
      checked::CreateResult r;
      if(raw.n_elem==14&&cp.n_elem==6){std::array<double,14>a{};std::copy(raw.begin(),raw.end(),a.begin());r=checked::try_create_gcamp(a,{cp[0],cp[1],cp[2],cp[3],cp[4],cp[5]},c);}
      py::object model=py::none();if(r.model)model=py::cast(std::move(r.model));return py::make_tuple(r.report,model);
    });
    gcamp.def("try_set_params",[](GCaMP& x,double a,double b,double c,double d,double e,double f,const checked::Context& ctx){return x.try_set_params({a,b,c,d,e,f},ctx).report;})
      .def("try_set_state",[](GCaMP& x,double ca,const checked::Context& ctx){return x.try_set_state(ca,ctx).report;})
      .def("try_init",[](GCaMP& x,const checked::Context& ctx){return x.try_init(ctx).report;})
      .def("try_set_params_and_init",[](GCaMP& x,double a,double b,double c,double d,double e,double f,const checked::Context& ctx){return x.try_set_params_and_init({a,b,c,d,e,f},ctx).report;})
      .def("try_steady_state",[](GCaMP& x,double ca,const checked::Context& ctx){auto r=x.try_steady_state(ca,ctx);py::object state=py::none();if(r.state){arma::vec v(12);std::copy(r.state->begin(),r.state->end(),v.begin());state=py::cast(v);}return py::make_tuple(r.report,state);});
#ifdef C_SPIKES_CHECKED_TESTING
    m.def("_checked_seed_histories",&checked::TestAccess::seed);
    m.def("_checked_snapshot",[](const GCaMP& x){return py::bytes(checked::TestAccess::snapshot(x));});
    m.def("_checked_fault",[](int call,Status status){checked::testing::reset();checked::testing::fault().call=call;checked::testing::fault().status=status;});
    m.def("_checked_clear_fault",&checked::testing::reset);
#endif
#endif
    // Add Kokkos cleanup callback	
    m.add_object("_cleanup", py::capsule(cleanup_callback));
}
