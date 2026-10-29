#ifndef GCAMPMODEL_H
#define GCAMPMODEL_H
#include<armadillo>
#include "numerical_diagnostics.h"
#ifdef C_SPIKES_CHECKED_CPU
#include "checked_equilibrium.h"
#endif

using namespace std;

enum TimeStepMode {FIXED, FIXEDLA};

struct GCaMP_params {
    
    double Gparams[14];

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

};


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
		GCaMP(const arma::vec Gparams_in, const arma::vec Cparams_in);
		
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
    double fixedStep_LA_threadsafe(double deltat, int ns, const arma::vec& state_in, arma::vec& state_out);
    void evolve(double deltat, int ns, const arma::vec& state);
    void evolve_threadsafe(double deltat, int ns, const arma::vec& state_in, arma::vec& state_out, double & dff_out);

    arma::vec steady_state(double c0);
    void init();
    double flux(double ca, const arma::vec& G);
    double flux_konN_konC(double konN, double konC, const arma::vec& G);
    void setGmat(double ca);
    void fillGmat(arma::mat::fixed<9,9> & Gmatrix, double ca);
    void setGmat_konN_konC(arma::mat::fixed<9,9> & Gmatrix, double konN, double konC);
    void setState(double ca);
    void setTimeStepMode(TimeStepMode);
    double getDFF();
    double getDFF(const arma::vec&);
    double getAmplitude();

    arma::vec::fixed<12> state;
    arma::vec::fixed<12> initial_state;
    double DFF;
    
    arma::vec::fixed<14> Gparams;
		arma::vec::fixed<6> Cparams;
    arma::mat::fixed<9,9> Gmat;
    arma::uvec::fixed<5> brightStates;

    int step_count=0;

    // Adaptive time step
    TimeStepMode TSMode=FIXED;
		
		// Methods for making and retrieving GCaMP simulations via python bindings
		void integrateOverTime(const arma::vec& time_vect, const arma::vec& spike_times);
		void integrateOverTime2(const arma::vec& time_vect, const arma::vec& spike_times);
		
    // Method to retrieve the stored DFF and state values
    //const arma::vec& getDFFValues() const;
    arma::vec getDFFValues() const { csp_diagnostics::Scope diagnostic("GCaMP.getDFFValues"); diagnostic.mode(TSMode); diagnostic.array(DFF_values, "DFF_values", csp_diagnostics::output_value); return DFF_values; }
    arma::mat getGValues() const { csp_diagnostics::Scope diagnostic("GCaMP.getGValues"); diagnostic.mode(TSMode); diagnostic.array(G_interp, "G_interp", csp_diagnostics::output_value); return G_interp; }
    arma::vec getBCaValues() const { csp_diagnostics::Scope diagnostic("GCaMP.getBCaValues"); diagnostic.mode(TSMode); diagnostic.array(BCa_values, "BCa_values", csp_diagnostics::output_value); return BCa_values; }
    arma::vec getCaValues() const { csp_diagnostics::Scope diagnostic("GCaMP.getCaValues"); diagnostic.mode(TSMode); diagnostic.array(Ca_values, "Ca_values", csp_diagnostics::output_value); return Ca_values; }
    arma::vec getCaInValues() const { csp_diagnostics::Scope diagnostic("GCaMP.getCaInValues"); diagnostic.mode(TSMode); diagnostic.array(Ca_in_values, "Ca_in_values", csp_diagnostics::output_value); return Ca_in_values; }
    
    // method required for kokkos
    void read_params(GCaMP_params & params);

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
		arma::vec DFF_values;
    arma::mat G_values;       // For the full-length simulation time
    arma::mat G_interp;       // For interpolating onto the experimental time vector
    arma::vec BCa_values;
    arma::vec Ca_values;
    arma::vec Ca_in_values;

};

#endif
