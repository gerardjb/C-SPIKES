#ifndef GCAMPMODEL_H
#define GCAMPMODEL_H
#include <string>
#include "generator9.h"
#include "scalar_input_view.h"
#include "scalar_output_view.h"
#include "state12.h"
#include "parameter_block.h"
#include "bright_indices5.h"
#include "history_storage.h"
#include "numerical_diagnostics.h"
#ifdef C_SPIKES_CHECKED_CPU
#include "checked_equilibrium.h"
#endif

using namespace std;

enum TimeStepMode {FIXED, FIXEDLA};

#include "GCaMP_params.h"


#ifdef CSP_G1C_TEST_ACCESS
namespace g1c { struct Access; }
#endif
class GCaMP {
#ifdef C_SPIKES_CHECKED_CPU
    friend struct checked::Engine;
#ifdef C_SPIKES_CHECKED_TESTING
    friend struct checked::TestAccess;
#endif
    struct CheckedTag {};
    explicit GCaMP(CheckedTag) {}
#endif
#ifdef CSP_G1C_TEST_ACCESS
    friend struct g1c::Access;
#endif
public:
    GCaMP(double G_tot,double gamma, double DCaT, double Rf, double gam_in, double gam_out, string Gparam_file="");
		GCaMP(string Gparam_file, string Cparam_file);
		GCaMP(csp_model::ScalarInputView indicator_input, csp_model::ScalarInputView cell_input);
		
		void initial_setup();
    void setParams(double G_tot,double gamma, double DCaT, double Rf, double gam_in, double gam_out);
#ifdef C_SPIKES_CHECKED_CPU
    checked::UpdateResult try_set_params(checked::CellParameters,const checked::Context&);
    checked::UpdateResult try_set_state(double,const checked::Context&);
    checked::UpdateResult try_init(const checked::Context&);
    checked::UpdateResult try_set_params_and_init(checked::CellParameters,const checked::Context&);
    checked::EquilibriumResult try_steady_state(double,const checked::Context&);
#endif
    
    void evolve(double deltat, int ns);
    void fixedStep(double deltat, int ns);
    void fixedStep_LA(double deltat, int ns);
    double fixedStep_LA_threadsafe(double deltat, int ns, csp_model::ScalarInputView state_in, csp_model::ScalarOutputView state_out);
    void evolve(double deltat, int ns, csp_model::ScalarInputView state);
    void evolve_threadsafe(double deltat, int ns, csp_model::ScalarInputView state_in, csp_model::ScalarOutputView state_out, double & dff_out);

    csp_model::State12 steady_state(double c0);
    void init();
    double flux(double ca, csp_model::ScalarInputView G);
    double flux_konN_konC(double konN, double konC, csp_model::ScalarInputView G);
    void setGmat(double ca);
    void fillGmat(csp_model::Generator9 & Gmatrix, double ca);
    void setGmat_konN_konC(csp_model::Generator9 & Gmatrix, double konN, double konC);
    void setState(double ca);
    void setTimeStepMode(TimeStepMode);
    double getDFF();
    double getDFF(csp_model::ScalarInputView);
    double getAmplitude();

    csp_model::State12 state;
    csp_model::State12 initial_state;
    double DFF;
    
    csp_model::IndicatorParameters14 Gparams;
		csp_model::CellParameterRecord6 Cparams;
    csp_model::Generator9 Gmat;
    csp_model::BrightIndices5 brightStates{};

    int step_count=0;

    // Adaptive time step
    TimeStepMode TSMode=FIXED;
		
		// Methods for making and retrieving GCaMP simulations via python bindings
		void integrateOverTime(csp_model::ScalarInputView time_input, csp_model::ScalarInputView spike_input);
		
    // Method to retrieve the stored DFF and state values
    //const arma::vec& getDFFValues() const;
    csp_model::HistoryVector getDFFValues() const { CSP_OBSERVE(csp_diagnostics::Scope diagnostic("GCaMP.getDFFValues");) CSP_OBSERVE(diagnostic.mode(TSMode);) CSP_OBSERVE(diagnostic.array(csp_model::scan_view(DFF_values), "DFF_values", csp_diagnostics::output_value);) return DFF_values; }
    csp_model::HistoryMatrix getGValues() const { CSP_OBSERVE(csp_diagnostics::Scope diagnostic("GCaMP.getGValues");) CSP_OBSERVE(diagnostic.mode(TSMode);) CSP_OBSERVE(diagnostic.array(csp_model::scan_view(G_interp), "G_interp", csp_diagnostics::output_value);) return G_interp; }
    csp_model::HistoryVector getBCaValues() const { CSP_OBSERVE(csp_diagnostics::Scope diagnostic("GCaMP.getBCaValues");) CSP_OBSERVE(diagnostic.mode(TSMode);) CSP_OBSERVE(diagnostic.array(csp_model::scan_view(BCa_values), "BCa_values", csp_diagnostics::output_value);) return BCa_values; }
    csp_model::HistoryVector getCaValues() const { CSP_OBSERVE(csp_diagnostics::Scope diagnostic("GCaMP.getCaValues");) CSP_OBSERVE(diagnostic.mode(TSMode);) CSP_OBSERVE(diagnostic.array(csp_model::scan_view(Ca_values), "Ca_values", csp_diagnostics::output_value);) return Ca_values; }
    csp_model::HistoryVector getCaInValues() const { CSP_OBSERVE(csp_diagnostics::Scope diagnostic("GCaMP.getCaInValues");) CSP_OBSERVE(diagnostic.mode(TSMode);) CSP_OBSERVE(diagnostic.array(csp_model::scan_view(Ca_in_values), "Ca_in_values", csp_diagnostics::output_value);) return Ca_in_values; }
    
    // method required for kokkos
    void read_params(GCaMP_params & params);

    std::array<double,2> getPoolTotals() const { return {{G_tot, B_tot}}; }

private:
    // parameters that are allowed to vary
    double G_tot;
    double gamma;
    double DCaT;
    double Rf;

    // parameters that can be fixed
    // indicator 
    double konN,   koffN;
		double konC,   koffC;
		double konPN,  koffPN;
		double konPN2, koffPN2;
		double konPC,  koffPC;
		double konPC2,  koffPC2;

    //calcium
    const double c0     = 5e-8;
    const double FWHM   = 2.8e-4;
    double sigma2_calcium_spike;
    double gam_in;
    double gam_out;
    
    // buffer
    const double koff_B = 1e4;
    const double kon_B  = 1e8;
    const double B_tot  = 0.004;
    double BCa0;
    double kapB;

    // fluorescence
    const double csat   = 1e-2;

    // dFF normalization
    double G0, Gsat, Ginit;
		
		// For python output
		csp_model::HistoryVector DFF_values;
    csp_model::HistoryMatrix G_values;       // For the full-length simulation time
    csp_model::HistoryMatrix G_interp;       // For interpolating onto the experimental time vector
    csp_model::HistoryVector BCa_values;
    csp_model::HistoryVector Ca_values;
    csp_model::HistoryVector Ca_in_values;

};

#endif
