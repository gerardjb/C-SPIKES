#include "include/application_numpy_bridge.h"
//Required pybind libraries for expected conversions
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//External libraries called by pgas
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <json/json.h>
#include <fstream>
#include <iostream>
#include "include/scalar_input_numpy_bridge.h"
#include "include/history_numpy_bridge.h"
#include "include/equilibrium_numpy_bridge.h"
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
        .def(py::init([](const csp_python::ScalarInputCopy& time, const csp_python::ScalarInputCopy& data,
                      const std::string& constants_file, const std::string& output_folder, unsigned int column,
                      const std::string& tag, unsigned int niter, const std::string& trainedPriorFile,
                      bool append, unsigned int trim, bool verbose, const csp_python::ScalarInputCopy& gtSpikes,
                      bool has_trained_priors, bool has_gtspikes, unsigned int maxlen, const std::string& Gparam_file, int seed) {
             return std::make_unique<Analyzer>(time.values, data.values, constants_file, output_folder,
                      column, tag, niter, trainedPriorFile, append, trim, verbose, gtSpikes.values,
                      has_trained_priors, has_gtspikes, maxlen, Gparam_file, seed);
         }),
             py::arg("time"), py::arg("data"), py::arg("constants_file"), py::arg("output_folder"), py::arg("column"), py::arg("tag"),
             py::arg("niter") = 0, py::arg("trainedPriorFile") = "", py::arg("append") = false, py::arg("trim") = 1,
             py::arg("verbose") = true, py::arg("gtSpikes") = 0, py::arg("has_trained_priors") = false, py::arg("has_gtspikes") = false,
             py::arg("maxlen") = 0, py::arg("Gparam_file") = "", py::arg("seed") = 0)
        .def("run", &Analyzer::run, py::call_guard<csp_diagnostics::PythonScope>())
        .def("get_run_status", &Analyzer::get_run_status)
        .def("get_run_message", &Analyzer::get_run_message)
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
				.def(py::init([](const csp_python::ScalarInputCopy& g, const csp_python::ScalarInputCopy& c){ return std::make_unique<GCaMP>(g.view(),c.view()); }), py::arg("Gparams_in"), py::arg("Cparams_in"), py::call_guard<csp_diagnostics::PythonScope>())
        .def("setParams", &GCaMP::setParams, py::call_guard<csp_diagnostics::PythonScope>())
        .def("setGmat", &GCaMP::setGmat, py::call_guard<csp_diagnostics::PythonScope>())
        .def("flux", [](GCaMP& g, double ca, const csp_python::ScalarInputCopy& x){return g.flux(ca,x.view());}, py::call_guard<csp_diagnostics::PythonScope>())
        .def("steady_state", [](GCaMP& model, double ca) -> csp_python::EquilibriumOutput12 { const auto result = model.steady_state(ca); return {result.coefficients}; }, py::call_guard<csp_diagnostics::PythonScope>())
        .def("init", &GCaMP::init, py::call_guard<csp_diagnostics::PythonScope>())
        .def("setState", &GCaMP::setState, py::call_guard<csp_diagnostics::PythonScope>())
        .def("setTimeStepMode", &GCaMP::setTimeStepMode, py::arg("mode").noconvert(), py::call_guard<csp_diagnostics::PythonScope>())
        .def("evolve", [](GCaMP& g, double dt, int ns, const csp_python::ScalarInputCopy& x){g.evolve(dt,ns,x.view());}, py::call_guard<csp_diagnostics::PythonScope>())
        .def("evolve", py::overload_cast<double, int>(&GCaMP::evolve), py::call_guard<csp_diagnostics::PythonScope>())
        .def("fixedStep", &GCaMP::fixedStep, py::call_guard<csp_diagnostics::PythonScope>())
        .def("fixedStep_LA", &GCaMP::fixedStep_LA, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getDFF", py::overload_cast<>(&GCaMP::getDFF), py::call_guard<csp_diagnostics::PythonScope>())
        .def("getDFF", [](GCaMP& g, const csp_python::ScalarInputCopy& x){return g.getDFF(x.view());}, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getAmplitude", &GCaMP::getAmplitude, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getPoolTotals", [](const GCaMP &g) {
            const auto totals = g.getPoolTotals();
            py::dict result; result["G_tot_M"] = totals[0]; result["B_tot_M"] = totals[1];
            return result;
        })
        .def("integrateOverTime", [](GCaMP& g, const csp_python::ScalarInputCopy& t, const csp_python::ScalarInputCopy& spikes){g.integrateOverTime(t.view(),spikes.view());}, py::arg("time"), py::arg("spike_times"), py::call_guard<csp_diagnostics::PythonScope>())
        .def("getDFFValues", [](const GCaMP& g){return csp_python::numpy_copy(g.getDFFValues());}, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getGValues", [](const GCaMP& g){return csp_python::numpy_copy(g.getGValues());}, py::call_guard<csp_diagnostics::PythonScope>())
        .def("getStates", [](const GCaMP &g) {
            csp_diagnostics::Scope numerical_states("GCaMP.getStates");
            py::dict states;
            states["G"] = csp_python::numpy_copy(g.getGValues());
            states["BCa"] = csp_python::numpy_copy(g.getBCaValues());
            states["Ca"] = csp_python::numpy_copy(g.getCaValues());
            states["Ca_in"] = csp_python::numpy_copy(g.getCaInValues());
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
    m.def("try_create_gcamp",[](const csp_python::ScalarInputCopy& raw_owned,const csp_python::ScalarInputCopy& cell_owned,const checked::Context& c){
      const auto raw=raw_owned.view();const auto cp=cell_owned.view();
      checked::CreateResult r;
      if(raw.size()==14&&cp.size()==6){std::array<double,14>a{};std::copy(raw.data(),raw.data()+14,a.begin());r=checked::try_create_gcamp(a,{cp(0),cp(1),cp(2),cp(3),cp(4),cp(5)},c);}
      py::object model=py::none();if(r.model)model=py::cast(std::move(r.model));return py::make_tuple(r.report,model);
    });
    gcamp.def("try_set_params",[](GCaMP& x,double a,double b,double c,double d,double e,double f,const checked::Context& ctx){return x.try_set_params({a,b,c,d,e,f},ctx).report;})
      .def("try_set_state",[](GCaMP& x,double ca,const checked::Context& ctx){return x.try_set_state(ca,ctx).report;})
      .def("try_init",[](GCaMP& x,const checked::Context& ctx){return x.try_init(ctx).report;})
      .def("try_set_params_and_init",[](GCaMP& x,double a,double b,double c,double d,double e,double f,const checked::Context& ctx){return x.try_set_params_and_init({a,b,c,d,e,f},ctx).report;})
      .def("try_steady_state",[](GCaMP& x,double ca,const checked::Context& ctx){auto r=x.try_steady_state(ca,ctx);py::object state=py::none();if(r.state){state=py::cast(csp_python::EquilibriumOutput12{*r.state});}return py::make_tuple(r.report,state);});
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
