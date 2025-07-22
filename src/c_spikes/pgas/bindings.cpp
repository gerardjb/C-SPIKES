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
// Attempting kokkos integration
#include <Kokkos_Core.hpp>
namespace py = pybind11;

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
	if (Kokkos::is_initialized()) Kokkos::finalize();
};


PYBIND11_MODULE(pgas_bound, m) {
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
        .def("run", &Analyzer::run)
        .def("add_parameter_sample", &Analyzer::add_parameter_sample)
        .def("get_parameter_samples", &Analyzer::get_parameter_samples);
		    //.def("get_final_params", &get_final_params, "Get final parameters as a NumPy array");
	
				
	// bindings for GCaMP_model.cpp			
		py::class_<GCaMP>(m, "GCaMP")
        .def(py::init<double, double, double, double, double, double, std::string>())
				.def(py::init<std::string, std::string>(), py::arg("Gparam_file"), py::arg("Cparam_file"))
				.def(py::init<const arma::vec,const arma::vec>(), py::arg("Gparams_in"), py::arg("Cparams_in"))
        .def("setParams", &GCaMP::setParams)
        .def("setGmat", &GCaMP::setGmat)
        .def("flux", &GCaMP::flux)
        .def("steady_state", &GCaMP::steady_state)
        .def("init", &GCaMP::init)
        .def("setState", &GCaMP::setState)
        .def("setTimeStepMode", &GCaMP::setTimeStepMode)
        .def("evolve", py::overload_cast<double, int, const arma::vec&>(&GCaMP::evolve))
        .def("evolve", py::overload_cast<double, int>(&GCaMP::evolve))
        .def("fixedStep", &GCaMP::fixedStep)
        .def("fixedStep_LA", &GCaMP::fixedStep_LA)
        .def("getDFF", py::overload_cast<>(&GCaMP::getDFF))
        .def("getDFF", py::overload_cast<const arma::vec&>(&GCaMP::getDFF))
        .def("getAmplitude", &GCaMP::getAmplitude)
        .def("integrateOverTime", &GCaMP::integrateOverTime, py::arg("time"), py::arg("spike_times"))
        .def("integrateOverTime2", &GCaMP::integrateOverTime2, py::arg("time"), py::arg("spike_times"))
        .def("getDFFValues", &GCaMP::getDFFValues)
        .def("getGValues", &GCaMP::getGValues)
        .def("getStates", [](const GCaMP &g) {
            py::dict states;
            states["G"] = g.getGValues();
            states["BCa"] = g.getBCaValues();
            states["Ca"] = g.getCaValues();
            states["Ca_in"] = g.getCaInValues();
            return states;
        });
    // Add Kokkos cleanup callback	
    m.add_object("_cleanup", py::capsule(cleanup_callback));
}