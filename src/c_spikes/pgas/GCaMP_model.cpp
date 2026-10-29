#include "include/model_algorithm_compat.h"
#include "include/model_backend.h"
#include "include/model_evolution_compat.h"
#include "include/legacy_equilibrium_compat.h"
#include <iostream>
#ifdef CSP_G1C_USE_GSL
#include <armadillo>
#endif
#include "include/brightness_reduction.h"
#include"include/GCaMP_model.h"
#include "include/numerical_diagnostics.h"
using namespace csp_diagnostics;
#ifdef CSP_G1C_USE_GSL
#include "model_trial.h"
#endif

using namespace std;

namespace {
void assign_rows(csp_model::Generator9& target,
                 const std::array<std::array<double,9>,9>& rows) noexcept {
    for (std::size_t row=0; row<9; ++row)
        for (std::size_t col=0; col<9; ++col)
            target(row,col)=rows[row][col];
}
#ifdef CSP_G1C_USE_GSL
arma::mat::fixed<9,9> owned_armadillo(const csp_model::Generator9& source) {
    arma::mat::fixed<9,9> result;
    for (std::size_t col=0; col<9; ++col)
        for (std::size_t row=0; row<9; ++row)
            result(row,col)=source(row,col);
    return result;
}
#endif
}


// Constructor used by the PGAS particles
GCaMP::GCaMP(double p1, double p2, double p3, double p4, double p5, double p6, string Gparam_file) {
    Scope diagnostic("GCaMP.__init__");
    diagnostic.mode(TSMode);
  // Set passed-in parameters
  G_tot = p1;
  gamma = p2;
  DCaT = p3;
  Rf = p4;
  gam_in = p5;
  gam_out = p6;

  // Load Gparams from file
  if (Gparam_file.empty()) {
    throw std::invalid_argument("GParam file cannot be empty");
  }
  csp_model::load_raw_ascii_compat(Gparams,Gparam_file);

  // Call initial_setup to complete object initialization
  initial_setup();
}

// Constructor for passing files
GCaMP::GCaMP(string Gparam_file, string Cparam_file) {
    Scope diagnostic("GCaMP.__init__");
    diagnostic.mode(TSMode);
	
  // Load Gparams from file or default values
  if (Gparam_file.empty()) {
    Gparams.coefficients = {
      4.0320349221094061E-7,
			74.94798544278855,
			1.7376760772790556,
			3.62856271417812E-8,
			0.6056939075038793,
			1.1024894240446206,
			30.699056808075316,
			311.52950002130206,
			0.15520110541159166,
			43.466019961996466,
			47.325861294367463,
			368.84356191597425,
			0.71415314204711966,
			31.860437304985595
    };
  } else {
    csp_model::load_raw_ascii_compat(Gparams,Gparam_file);
  }
	
	//Loading from cparams file (assumes order G_tot, gamma, DCaT, Rf, gam_in, gom_out
  if (Cparam_file.empty()) {
    // Use default values (as in the original constructor)
    G_tot = 1e-5;
    gamma = 1000;
    DCaT = 1e-5;
		Rf = 5;
		gam_in = 40;
		gam_out = 4;
  } else {
		csp_model::load_raw_ascii_compat(Cparams,Cparam_file);
		G_tot = Cparams(0);
    gamma = Cparams(1);
    DCaT = Cparams(2);
		Rf = Cparams(3);
		gam_in = Cparams(4);
		gam_out = Cparams(5);
	}
	
  // Call initial_setup to complete object initialization
  initial_setup();
}

// Passing in the parameters as numeric types
GCaMP::GCaMP(csp_model::ScalarInputView indicator_input, csp_model::ScalarInputView cell_input) {
    // Own both inputs for the complete construction; no borrowed storage escapes.
    const csp_model::workspace::NumericInputSnapshot Gparams_in(indicator_input.data(), indicator_input.size());
    const csp_model::workspace::NumericInputSnapshot Cparams_in(cell_input.data(), cell_input.size());
    Scope diagnostic("GCaMP.__init__");
    diagnostic.mode(TSMode);
  // Set member variables from passed-in arguments
	G_tot = Cparams_in(0);
  gamma = Cparams_in(1);
  DCaT = Cparams_in(2);
  Rf = Cparams_in(3);
  gam_in = Cparams_in(4);
  gam_out = Cparams_in(5);
  csp_model::workspace::assign_indicator14(Gparams,Gparams_in);  // Assuming Gparams is already a vector of appropriate size
	
	cout<<"G_tot = "<<G_tot<<"; gam_out = "<<gam_out<<endl;
	cout<<"G1 = "<<Gparams(0)<<"; G6 = "<<Gparams(5)<<endl;
  // Call initial_setup to complete object initialization
  initial_setup();
}


void GCaMP::initial_setup() {
    Scope diagnostic("GCaMP.initial_setup");
    diagnostic.mode(TSMode);
  brightStates = {2, 3, 5, 6, 8};
  const auto bright_indices = brightStates;
  diagnostic.array(Gparams, "Gparams", state_value);

  Gparams(0) = Gparams(1) / pow(Gparams(0), Gparams(2));
  Gparams(3) = Gparams(4) / pow(Gparams(3), Gparams(5));
	
	
  kapB = B_tot / (koff_B / kon_B);

  BCa0 = kon_B * c0 / (kon_B * c0 + koff_B) * B_tot;

  konN = Gparams(0) * pow(c0, Gparams(2)); koffN = Gparams(1);
  konC = Gparams(3) * pow(c0, Gparams(5)); koffC = Gparams(4);

  konPN = Gparams(6); koffPN = Gparams(7);
  konPC = Gparams(8); koffPC = Gparams(9);

  konPN2 = Gparams(10); koffPN2 = Gparams(11);
  konPC2 = Gparams(12); koffPC2 = Gparams(13);

  sigma2_calcium_spike = pow(FWHM, 2) / (8 * log(2));

  setState(c0);

  const auto G0_v = steady_state(0);
  const auto Gsat_v = steady_state(csat);
  const auto Ginit_v = steady_state(c0);

  G0 = csp_model::indexed_brightness_sum5(G0_v.data(), G0_v.n_elem, bright_indices);
  Gsat = csp_model::indexed_brightness_sum5(Gsat_v.data(), Gsat_v.n_elem, bright_indices);
  Ginit = csp_model::indexed_brightness_sum5(Ginit_v.data(), Ginit_v.n_elem, bright_indices);

  DFF = 0;
  diagnostic.array(state, "state", state_value);
  diagnostic.scalar(G0, "G0", state_value);
  diagnostic.scalar(Gsat, "Gsat", state_value);
  diagnostic.scalar(Ginit, "Ginit", state_value);
  diagnostic.scalar(Rf, "Rf", state_value);
}


void GCaMP::setParams(double p1, double p2, double p3, double p4, double p5, double p6){
    const auto bright_indices = brightStates;
    Scope diagnostic("GCaMP.setParams");
    diagnostic.mode(TSMode);
    G_tot = p1;
    gamma = p2;
    DCaT  = p3;
    Rf    = p4;
    gam_in = p5;
    gam_out = p6;

    const auto G0_v    = steady_state(0);
    const auto Gsat_v  = steady_state(csat);
    const auto Ginit_v = steady_state(c0);

    G0     =  csp_model::indexed_brightness_sum5(G0_v.data(), G0_v.n_elem, bright_indices);
    Gsat   =  csp_model::indexed_brightness_sum5(Gsat_v.data(), Gsat_v.n_elem, bright_indices);
    Ginit  =  csp_model::indexed_brightness_sum5(Ginit_v.data(), Ginit_v.n_elem, bright_indices);
    diagnostic.scalar(G0, "G0", state_value);
    diagnostic.scalar(Gsat, "Gsat", state_value);
    diagnostic.scalar(Ginit, "Ginit", state_value);
    diagnostic.scalar(Rf, "Rf", state_value);
   
}


void GCaMP::setGmat_konN_konC(csp_model::Generator9 & Gmatrix, double konN, double konC){
    Scope diagnostic("GCaMP.setGmat_konN_konC");
    diagnostic.equilibrium_context();

  Gmatrix(0, 0) = -(konN+konC);
  Gmatrix(1, 0) = konN;
  Gmatrix(1, 1) = -(koffN+konPN+konC);
  Gmatrix(2, 2) = -(konC+koffPN);
  Gmatrix(3, 2) = konC;
  Gmatrix(4, 0) = konC;
  Gmatrix(4, 4) = -(koffC+konPC+konN);
  Gmatrix(5, 5) = -(koffPC+konN);
  Gmatrix(6, 5) = konN;
  Gmatrix(7, 1) = konC;
  Gmatrix(7, 4) = konN;

    diagnostic.array(Gmatrix, "generator", rate_value);
}

void GCaMP::setGmat(double ca){
    Scope diagnostic("GCaMP.setGmat");
    diagnostic.equilibrium_context();

	konN   = Gparams(0)*pow(ca,Gparams(2));
  konC   = Gparams(3)*pow(ca,Gparams(5));

// note: because only 2 parameters need to be updated when changing calcium level, perhaps here we can optimize something

//	Gmat = {{-(konN+konC),koffN,koffPN,0,koffC,koffX2,0,0,0},
//            {konN,-(koffN+konPN+konC),0,0,0,0,koffPC,koffC,0},
//            {0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC},
//            {0,0,konC,-(koffC+konX+koffPN),0,0,0,konPN,0},
//            {konC,0,0,koffPN,-(koffC+konPC+konN),koffPC,0,koffN,0},
//            {0,0,0,0,konPC,-(koffPC+konN+koffX2),koffN,0,koffPN},
//            {0,0,0,0,0,konN,-(koffN+konPN+koffPC),konX,0},
//            {0,konC,0,0,konN,0,0,-(koffC+koffN+konX+konPN),0},
//            {0,0,0,konX,0,0,konPN,0,-(koffPN+koffPC)}};

	assign_rows(Gmat, {{{-(konN+konC),koffN,koffPN,0,koffC,koffPC,0,0,0},
		    {konN,-(koffN+konPN+konC),0,0,0,0,koffPC2,koffC,0},
			{0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC2},
			{0,0,konC,-(koffC+konPC2+koffPN2 ),0,0,0,konPN2,0},
			{konC,0,0,koffPN2,-(koffC+konPC+konN),0,0,koffN,0},
			{0,0,0,0,konPC,-(koffPC+konN),koffN,0,koffPN2},
			{0,0,0,0,0,konN,-(koffN+konPN2+koffPC2),konPC2,0},
			{0,konC,0,0,konN,0,0,-(koffC+koffN+konPC2+konPN2),0},
			{0,0,0,konPC2,0,0,konPN2,0,-(koffPN2+koffPC2)}}});

    diagnostic.array(Gmat, "generator", rate_value);
}

void GCaMP::fillGmat(csp_model::Generator9 & Gmatrix, double ca){
    Scope diagnostic("GCaMP.fillGmat");
    diagnostic.equilibrium_context();

	konN   = Gparams(0)*pow(ca,Gparams(2));
  konC   = Gparams(3)*pow(ca,Gparams(5));

	assign_rows(Gmatrix, {{{-(konN+konC),koffN,koffPN,0,koffC,koffPC,0,0,0},
		    {konN,-(koffN+konPN+konC),0,0,0,0,koffPC2,koffC,0},
			{0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC2},
			{0,0,konC,-(koffC+konPC2+koffPN2 ),0,0,0,konPN2,0},
			{konC,0,0,koffPN2,-(koffC+konPC+konN),0,0,koffN,0},
			{0,0,0,0,konPC,-(koffPC+konN),koffN,0,koffPN2},
			{0,0,0,0,0,konN,-(koffN+konPN2+koffPC2),konPC2,0},
			{0,konC,0,0,konN,0,0,-(koffC+koffN+konPC2+konPN2),0},
			{0,0,0,konPC2,0,0,konPN2,0,-(koffPN2+koffPC2)}}});

    diagnostic.array(Gmatrix, "generator", rate_value);
}

double GCaMP::flux(double ca, csp_model::ScalarInputView G){
    Scope diagnostic("GCaMP.flux");
    diagnostic.mode(TSMode);

	konN   = Gparams(0)*pow(ca,Gparams(2));
  konC   = Gparams(3)*pow(ca,Gparams(5));

  // Calculate fluxes per binding site
  double N = -(konN*(G(0) + G(4) + G(5))) +
                koffN*(G(1) + G(6)+ G(7))  + 
                koffPN*G(2)                + 
                koffPN2*(G(3) + G(8));

  double C = -(konC *(G(0) + G(1) + G(2))) +
                koffC*(G(4) + G(3) + G(7))  + 
                koffPC*(G(5))+ 
                koffPC2*(G(6)+G(8));

  int Csites=2;
  const double result = 2*N + Csites*C;
  diagnostic.scalar(result, "calcium_flux", rate_value);
  return result;
}

double GCaMP::flux_konN_konC(double konN, double konC, csp_model::ScalarInputView G){
    Scope diagnostic("GCaMP.flux_konN_konC");
    diagnostic.mode(TSMode);

  // Calculate fluxes per binding site
  double N = -(konN*(G(0) + G(4) + G(5))) +
                koffN*(G(1) + G(6)+ G(7))  + 
                koffPN*G(2)                + 
                koffPN2*(G(3) + G(8));

  double C = -(konC *(G(0) + G(1) + G(2))) +
                koffC*(G(4) + G(3) + G(7))  + 
                koffPC*(G(5))+ 
                koffPC2*(G(6)+G(8));

  int Csites=2;
  const double result = 2*N + Csites*C;
  diagnostic.scalar(result, "calcium_flux", rate_value);
  return result;
}



csp_model::State12 GCaMP::steady_state(double c0){
    Scope diagnostic("GCaMP.steady_state");
    diagnostic.equilibrium_context();
	setGmat(c0);
#ifdef CSP_G1C_USE_GSL
    // Explicit test-build opt-in only. The isolated owner handles failures
    // transactionally; default production builds retain Armadillo unchanged.
    const auto generator_bridge = owned_armadillo(Gmat);
    const auto candidate = g1c::solve(generator_bridge, G_tot);
    arma::vec ss(9);
    for (size_t i = 0; i < 9; ++i) ss(i) = candidate.state[i];
    arma::vec b_c = {BCa0,c0,c0};
    ss = arma::join_vert(ss, b_c);
    diagnostic.array(ss, "steady_state", state_value);
    csp_model::State12 result;
    for (std::size_t i = 0; i < 12; ++i) result[i] = ss(i);
#else
    const auto result = csp_model::legacy_equilibrium12_compat(Gmat, G_tot, BCa0, c0);
    diagnostic.array(result, "steady_state", state_value);
#endif
    return result;
}

void GCaMP::init(){
    Scope diagnostic("GCaMP.init");
    diagnostic.mode(TSMode);
    setState(c0);
    DFF=0;
}

void GCaMP::setState(double ca){
    Scope diagnostic("GCaMP.setState");
    diagnostic.mode(TSMode);
    const auto is = steady_state(ca);
    std::copy(is.begin(), is.end(), state.begin());
    std::copy(is.begin(), is.end(), initial_state.begin());
    diagnostic.array(state, "state", state_value);
}

void GCaMP::setTimeStepMode(TimeStepMode tsm){
    Scope diagnostic("GCaMP.setTimeStepMode");
    diagnostic.mode(TSMode);
    TSMode=tsm;
}

void GCaMP::evolve(double deltat, int ns, csp_model::ScalarInputView s){
    Scope diagnostic("GCaMP.evolve");
    diagnostic.mode(TSMode);
    diagnostic.array(csp_model::HistoryScan{s.data(),s.size()}, "supplied_state", state_value);
    if (s.size() != 12)
        throw std::logic_error("Mat::init(): size is fixed and hence cannot be changed");
    csp_model::State12 staged;
    for (std::size_t i = 0; i < 12; ++i) staged[i] = s(i);
    for (std::size_t i = 0; i < 12; ++i) state[i] = staged[i];
    evolve(deltat,ns);
}

void GCaMP::evolve(double deltat, int ns){
    Scope diagnostic("GCaMP.evolve");
    diagnostic.mode(TSMode);
    switch(TSMode){
        case FIXED:
            fixedStep(deltat,ns);
            break;
        case FIXEDLA:
            fixedStep_LA(deltat,ns);
            break;
    }
}

void GCaMP::evolve_threadsafe(double deltat, int ns, csp_model::ScalarInputView state_in, csp_model::ScalarOutputView state_out, double & dff_out){
    Scope diagnostic("GCaMP.evolve_threadsafe");
    diagnostic.mode(TSMode);
    switch(TSMode){
        case FIXED:
            fixedStep(deltat,ns);
            break;
        case FIXEDLA:
            dff_out = fixedStep_LA_threadsafe(deltat,ns,state_in,state_out);
            break;
    }
}


void GCaMP::fixedStep(double deltat, int ns){
    const auto bright_indices = brightStates;
    Scope diagnostic("GCaMP.fixedStep");
    diagnostic.mode(TSMode);
    diagnostic.array(state, "initial_state", state_value);
    diagnostic.scalar(Ginit-G0+(Gsat-G0)/(Rf-1), "fluorescence_denominator", normalization, -1, true);

    auto G = csp_model::workspace::indicator_copy(state);

    double BCa = state(9);
    double dBCa_dt;

    double Ca    = state(10);
    double Ca_in = state(11);
    double dCa_dt, dCa_in_dt;
    
    double Gflux;

    double fine_timestep=3e-6;
    double dt;

    auto grid_storage = csp_model::workspace::positive_grid_compat(0,fine_timestep,deltat);
    const auto& timesteps = grid_storage;
    auto calcium_input = csp_model::workspace::gaussian_input_compat(
        {timesteps.data(),timesteps.size()}, deltat, DCaT, sigma2_calcium_spike);
    
    for(unsigned int i=1;i<timesteps.size();++i){
        diagnostic.step(TSMode, "FIXED", deltat, fine_timestep, i, Ca, BCa, Ca_in);
        setGmat(Ca);
        csp_model::workspace::Indicator9 dG_dt;
        csp_model::workspace::product9_compat(Gmat, G, dG_dt);
        Gflux = flux(Ca,{G.memptr(),G.n_elem});

        dBCa_dt = kon_B*(B_tot-BCa)*Ca - koff_B*BCa;
        dCa_dt  = -gamma*(Ca-c0)  
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0)  //intra-compartmental exchange
            + Gflux - dBCa_dt + ns*calcium_input(i);
        dCa_in_dt = gam_in*(Ca - c0) - gam_out*(Ca_in - c0);

        dt  = timesteps(i)-timesteps(i-1);
        
        csp_model::workspace::advance9_compat(G, dt, dG_dt);
        BCa = BCa +  dt*dBCa_dt;
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in+= dt*dCa_in_dt;
        diagnostic.array(G, "G", state_value);
        diagnostic.scalar(BCa, "BCa", state_value, 9);
        diagnostic.scalar(Ca, "Ca", state_value, 10);
        diagnostic.scalar(Ca_in, "Ca_in", state_value, 11);


        step_count+=1;
    }

    for(unsigned int i=0;i<9;i++) state(i) = G(i);
    state(9)  = BCa;
    state(10) = Ca;

    DFF = (csp_model::indexed_brightness_sum5(G.memptr(), G.n_elem, bright_indices) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));                                
    diagnostic.scalar(DFF, "DFF", output_value);
}

void GCaMP::fixedStep_LA(double deltat, int ns){
    const auto bright_indices = brightStates;
    Scope diagnostic("GCaMP.fixedStep_LA");
    diagnostic.mode(TSMode);
    diagnostic.array(state, "initial_state", state_value);
    diagnostic.scalar(Ginit-G0+(Gsat-G0)/(Rf-1), "fluorescence_denominator", normalization, -1, true);

    auto G = csp_model::workspace::indicator_copy(state);

    double BCa = state(9);
    double dBCa_dt;

    double Ca  = state(10);
    double Ca_in = state(11);
    double dCa_dt, dCa_in_dt;
    
    double Gflux, Cflux;
    double finedt=100e-6;
    double dt;

    auto grid_storage = csp_model::workspace::positive_grid_compat(0,finedt,deltat);
    const auto& timesteps = grid_storage;
    double calcium_input;
    
    for(unsigned int i=1;i<timesteps.size();++i){
        diagnostic.step(TSMode, "FIXEDLA", deltat, finedt, i, Ca, BCa, Ca_in);

        calcium_input = (i==1) ? ns*DCaT/finedt : 0;

        double logCa = log(Ca);
	      double konN = Gparams(0)*exp(Gparams(2)*logCa);
        double konC = Gparams(3)*exp(Gparams(5)*logCa);

        setGmat_konN_konC(Gmat, konN, konC);
        csp_model::workspace::Indicator9 dG_dt;
        csp_model::workspace::product9_compat(Gmat, G, dG_dt);
        Gflux = flux_konN_konC(konN, konC, {G.memptr(),G.n_elem});

        Cflux   = -gamma*(Ca-c0) + Gflux;
        dBCa_dt = Cflux*kapB/(kapB + 1);
        dCa_in_dt = gam_in*(Ca-c0) - gam_out*(Ca_in - c0);

        dCa_dt  = -gamma*(Ca-c0) // pump out
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input*1/(kapB + 1);

        //cout<<timesteps(i)<<endl;
        //cout<<"BCa = "<<BCa<<"; d_dt:"<<dBCa_dt<<", "<<calcium_input*kapB/(kapB+1)<<endl;
        //cout<<"Ca  = "<<Ca<<" ; d_dt:"<<dCa_dt<<", "<<calcium_input*1/(kapB+1)<<endl;
        //cout<<"------------------------"<<endl;

        dt  = timesteps(i)-timesteps(i-1); 
        csp_model::workspace::advance9_compat(G, dt, dG_dt);
        BCa = BCa +  dt*(dBCa_dt+kapB/(kapB+1)*calcium_input);
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in = Ca_in + dt*dCa_in_dt;
        diagnostic.array(G, "G", state_value);
        diagnostic.scalar(BCa, "BCa", state_value, 9);
        diagnostic.scalar(Ca, "Ca", state_value, 10);
        diagnostic.scalar(Ca_in, "Ca_in", state_value, 11);
    }

    for(unsigned int i=0;i<9;i++) state(i) = G(i);
    state(9)  = BCa;
    state(10) = Ca;
    state(11) = Ca_in;

    DFF = (csp_model::indexed_brightness_sum5(G.memptr(), G.n_elem, bright_indices) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));                                
    diagnostic.scalar(DFF, "DFF", output_value);
}

double GCaMP::fixedStep_LA_threadsafe(double deltat, int ns, csp_model::ScalarInputView state_in, csp_model::ScalarOutputView state_out){
    Scope diagnostic("GCaMP.fixedStep_LA_threadsafe");
    diagnostic.mode(TSMode);
    diagnostic.array(csp_model::HistoryScan{state_in.data(),state_in.size()}, "initial_state", state_value);
    diagnostic.scalar(Ginit-G0+(Gsat-G0)/(Rf-1), "fluorescence_denominator", normalization, -1, true);

    if (state_in.size() < 9)
        throw std::out_of_range("Col::subvec(): indices out of bounds or incorrectly used");
    csp_model::workspace::Indicator9 G;
    for (std::size_t i = 0; i < 9; ++i) G(i) = state_in(i);
    csp_model::Generator9 Gmatrix;
    
    double BCa = state_in(9);
    double dBCa_dt;

    double Ca  = state_in(10);
    double Ca_in = state_in(11);
    double dCa_dt, dCa_in_dt;
    
    double Gflux, Cflux;
    double finedt=100e-6;
    double dt;

    auto grid_storage = csp_model::workspace::positive_grid_compat(0,finedt,deltat);
    const auto& timesteps = grid_storage;
    double calcium_input;

    // Intiliaze the Gmatrix
    fillGmat(Gmatrix, Ca);

    for(unsigned int i=1;i<timesteps.size();++i){
        diagnostic.step(TSMode, "FIXEDLA", deltat, finedt, i, Ca, BCa, Ca_in);

        calcium_input = (i==1) ? ns*DCaT/finedt : 0;

        double logCa = log(Ca);
	      double konN = Gparams(0)*exp(Gparams(2)*logCa);
        double konC = Gparams(3)*exp(Gparams(5)*logCa);

        setGmat_konN_konC(Gmatrix, konN, konC);
        csp_model::workspace::Indicator9 dG_dt;
        csp_model::workspace::product9_compat(Gmatrix, G, dG_dt);
        Gflux = flux_konN_konC(konN, konC, {G.memptr(),G.n_elem});

        Cflux   = -gamma*(Ca-c0) + Gflux;
        dBCa_dt = Cflux*kapB/(kapB + 1);
        dCa_in_dt = gam_in*(Ca-c0) - gam_out*(Ca_in - c0);

        dCa_dt  = -gamma*(Ca-c0) // pump out
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input*1/(kapB + 1);

        //cout<<timesteps(i)<<endl;
        //cout<<"BCa = "<<BCa<<"; d_dt:"<<dBCa_dt<<", "<<calcium_input*kapB/(kapB+1)<<endl;
        //cout<<"Ca  = "<<Ca<<" ; d_dt:"<<dCa_dt<<", "<<calcium_input*1/(kapB+1)<<endl;
        //cout<<"------------------------"<<endl;

        dt  = timesteps(i)-timesteps(i-1); 
        csp_model::workspace::advance9_compat(G, dt, dG_dt);
        BCa = BCa +  dt*(dBCa_dt+kapB/(kapB+1)*calcium_input);
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in = Ca_in + dt*dCa_in_dt;
        diagnostic.array(G, "G", state_value);
        diagnostic.scalar(BCa, "BCa", state_value, 9);
        diagnostic.scalar(Ca, "Ca", state_value, 10);
        diagnostic.scalar(Ca_in, "Ca_in", state_value, 11);
    }

    for(unsigned int i=0;i<9;i++) state_out(i) = G(i);
    state_out(9)  = BCa;
    state_out(10) = Ca;
    state_out(11) = Ca_in;

    //double DFF_out = (arma::accu(G(brightStates)) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));<- avoids loading arma to gpu
    double brightStatesSum = 0.0;
    for(unsigned int i=0;i<5;i++) brightStatesSum += G[brightStates[i]];
    double DFF_out = (brightStatesSum - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));

    diagnostic.scalar(DFF_out, "DFF", output_value);
    return DFF_out;
}


// This method allows direct outputs to python via bindings
void GCaMP::integrateOverTime(csp_model::ScalarInputView time_input, csp_model::ScalarInputView spike_input) {
    // Strict read-only descriptors preserve live native alias reads. These local
    // const objects never write, resize, own or retain the borrowed input storage.
    const auto time_vect = time_input;
    const auto spike_times = spike_input;
    const auto bright_indices = brightStates;
    Scope diagnostic("GCaMP.integrateOverTime");
    diagnostic.mode(TSMode);
    diagnostic.array(state, "initial_state", state_value);
    diagnostic.scalar(Ginit-G0+(Gsat-G0)/(Rf-1), "fluorescence_denominator", normalization, -1, true);
    // Clear previous state and DFF values
		DFF_values.reset();
    G_values.reset();
    BCa_values.reset();
    Ca_values.reset();
    Ca_in_values.reset();
		
		double fine_dt = 100e-6; // Hard code this near instability edge for hardest G/Cparams

		auto G = csp_model::workspace::indicator_copy(state);
    double BCa = state(9);
    double Ca = state(10);
    double Ca_in = state(11);
    double dBCa_dt, dCa_dt, dCa_in_dt, Gflux, Cflux;

    // Generate timesteps, prep calcium vect

    // regspace will not be inclusize of the last time value, we need to go one increment further so that we
    // can interpolate later.
    double t_end = time_vect(time_vect.size() - 1);
    double t_start = time_vect(0);
    int M = (int)floor((t_end-t_start)/fine_dt);
    double new_t_end = t_start+(M+1)*fine_dt;

//    arma::vec timesteps = arma::regspace(time_vect(0), fine_dt, time_vect(time_vect.size() - 1));
    auto grid_storage = csp_model::workspace::positive_grid_compat(time_vect(0), fine_dt, new_t_end);
    // Preserve the existing absolute lattice and its prefix; add only missing support.
    while (grid_storage(grid_storage.size() - 1) < t_end) {
        const std::size_t next_index = grid_storage.size();
        csp_model::HistoryVector extended_grid;
        extended_grid.set_size(next_index + 1);
        for (std::size_t j = 0; j < next_index; ++j) extended_grid(j) = grid_storage(j);
        extended_grid(next_index) = t_start + static_cast<double>(next_index) * fine_dt;
        grid_storage = std::move(extended_grid);
    }
    const auto& timesteps = grid_storage;
    size_t num_steps = timesteps.size();
    csp_model::HistoryVector calcium_input;
    calcium_input.set_size(timesteps.size());
    for (std::size_t j=0; j<calcium_input.size(); ++j) calcium_input(j)=0.;

		// Temporary vector to store intermediate DFF values
    auto temp_DFF_values = csp_model::workspace::zeroed_temporary_compat(timesteps.size());
    int spike_counter=0;
    // Find all timesteps that are within fine_timestep of one of the elements of spike_times
    for (std::size_t spike_index=0; spike_index<spike_times.size(); ++spike_index) {
        const double spike_time = spike_times(spike_index);
        const auto count = csp_model::workspace::fill_spike_matches(calcium_input, grid_storage, spike_time, fine_dt/2, DCaT/fine_dt);
        spike_counter+=count;
    }

    // Preallocate state vectors
    G_values.set_size(num_steps, G.n_elem);
    BCa_values.set_size(num_steps);
    Ca_values.set_size(num_steps);
    Ca_in_values.set_size(num_steps);

    // Initialize first state
    csp_model::workspace::copy_indicator_row_compat(G_values, 0, G);
    BCa_values(0) = BCa;
    Ca_values(0) = Ca;
    Ca_in_values(0) = Ca_in;
    temp_DFF_values(0) = (csp_model::indexed_brightness_sum5(G.memptr(), G.n_elem, bright_indices) - Ginit) /
                         (Ginit - G0 + (Gsat - G0) / (Rf - 1));

    for (unsigned int i = 1; i < timesteps.size(); ++i) {
        diagnostic.step(TSMode, "integrateOverTime", t_end-t_start, fine_dt, i, Ca, BCa, Ca_in);
				
				/*
					Giovanni's implementation
				*/
				setGmat(Ca);
				csp_model::workspace::Indicator9 dG_dt;
        csp_model::workspace::product9_compat(Gmat, G, dG_dt);
				Gflux = flux(Ca,{G.memptr(),G.n_elem});
				Cflux   = -gamma*(Ca-c0) + Gflux;
        dBCa_dt = Cflux*kapB/(kapB + 1);
        dCa_in_dt = gam_in*(Ca-c0) - gam_out*(Ca_in - c0);

        dCa_dt  = -gamma*(Ca-c0) // pump out
            -gam_in*(Ca - c0) + gam_out*(Ca_in - c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input(i)*1/(kapB + 1);
						
				csp_model::workspace::advance9_compat(G, fine_dt, dG_dt);
        BCa = BCa +  fine_dt*(dBCa_dt+kapB/(kapB+1)*calcium_input(i));
        Ca  = Ca  +  fine_dt*dCa_dt;
        Ca_in = Ca_in + fine_dt*dCa_in_dt;
        diagnostic.array(G, "G", state_value);
        diagnostic.scalar(BCa, "BCa", state_value, 9);
        diagnostic.scalar(Ca, "Ca", state_value, 10);
        diagnostic.scalar(Ca_in, "Ca_in", state_value, 11);

        // Store DFF value at each timestep
        temp_DFF_values(i) = (csp_model::indexed_brightness_sum5(G.memptr(), G.n_elem, bright_indices) - Ginit) /
                             (Ginit - G0 + (Gsat - G0) / (Rf - 1));
        
        // Store states
        csp_model::workspace::copy_indicator_row_compat(G_values, i, G);
        BCa_values(i) = BCa;
        Ca_values(i) = Ca;
        Ca_in_values(i) = Ca_in;
    }

    // Interpolate back onto original time vector
    {
        auto dff_history_bridge = csp_model::linear_interpolate_compat(
            {timesteps.data(),timesteps.size()}, {temp_DFF_values.data(),temp_DFF_values.size()},
            {time_vect.data(),time_vect.size()});
        DFF_values.set_size(dff_history_bridge.size());
        for(std::size_t i=0;i<DFF_values.size();++i)DFF_values(i)=dff_history_bridge(i);
    }
    {
        const auto source_history = BCa_values;
        auto history_bridge = csp_model::linear_interpolate_compat(
            {timesteps.data(),timesteps.size()}, {source_history.data(),source_history.size()},
            {time_vect.data(),time_vect.size()});
        BCa_values.set_size(history_bridge.size());
        for(std::size_t i=0;i<BCa_values.size();++i)BCa_values(i)=history_bridge(i);
    }
    {
        const auto source_history = Ca_values;
        auto history_bridge = csp_model::linear_interpolate_compat(
            {timesteps.data(),timesteps.size()}, {source_history.data(),source_history.size()},
            {time_vect.data(),time_vect.size()});
        Ca_values.set_size(history_bridge.size());
        for(std::size_t i=0;i<Ca_values.size();++i)Ca_values(i)=history_bridge(i);
    }
    {
        const auto source_history = Ca_in_values;
        auto history_bridge = csp_model::linear_interpolate_compat(
            {timesteps.data(),timesteps.size()}, {source_history.data(),source_history.size()},
            {time_vect.data(),time_vect.size()});
        Ca_in_values.set_size(history_bridge.size());
        for(std::size_t i=0;i<Ca_in_values.size();++i)Ca_in_values(i)=history_bridge(i);
    }
    // Now the same for G
    // Interpolate G_values for each G element
    size_t num_G_elements = G_values.cols();
    G_interp.set_size(time_vect.size(), num_G_elements);
    for (size_t col = 0; col < num_G_elements; ++col) {
        csp_model::HistoryVector G_col;
        G_col.set_size(G_values.rows());
        for(std::size_t row=0;row<G_col.size();++row)G_col(row)=G_values(row,col);
        auto G_interp_col = csp_model::linear_interpolate_compat(
            {timesteps.data(),timesteps.size()}, {G_col.data(),G_col.size()},
            {time_vect.data(),time_vect.size()});
        for(std::size_t row=0;row<G_interp.rows();++row)G_interp(row,col)=G_interp_col(row);
    }

    // For direct comparison of c++ to python stored values
    //cout<<"G_interp.row(0)"<<G_interp.row(0)<<endl;
    //cout<<"G_interp.row(200)"<<G_interp.row(200)<<endl;
    diagnostic.array(csp_model::scan_view(DFF_values), "DFF_history", output_value);
    diagnostic.array(csp_model::scan_view(G_interp), "G_history", output_value);
    diagnostic.array(csp_model::scan_view(BCa_values), "BCa_history", output_value);
    diagnostic.array(csp_model::scan_view(Ca_values), "Ca_history", output_value);
    diagnostic.array(csp_model::scan_view(Ca_in_values), "Ca_in_history", output_value);
}

//


// Giovanni's methods for passing compressed states around the SMC sampler
double GCaMP::getDFF(){
    const auto bright_indices = brightStates;
    Scope diagnostic("GCaMP.getDFF");
    diagnostic.mode(TSMode);
    const auto G = csp_model::gather_brightness5(state.data(), state.n_elem, bright_indices);
    const double result = (csp_model::sum_brightness5(G) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));
    diagnostic.scalar(Ginit-G0+(Gsat-G0)/(Rf-1), "fluorescence_denominator", normalization, -1, true);
    diagnostic.scalar(result, "DFF", output_value);
    return result;
}

double GCaMP::getDFF(csp_model::ScalarInputView s){
    const auto bright_indices = brightStates;
    Scope diagnostic("GCaMP.getDFF");
    diagnostic.mode(TSMode);
    // The input remains borrowed; the five-value gather is an owned snapshot.
    const auto G = csp_model::gather_brightness5(s.data(), s.size(), bright_indices);
    const double result = (csp_model::sum_brightness5(G) - Ginit)/(Ginit-G0+(Gsat-G0)/(Rf-1));
    diagnostic.scalar(Ginit-G0+(Gsat-G0)/(Rf-1), "fluorescence_denominator", normalization, -1, true);
    diagnostic.scalar(result, "DFF", output_value);
    return result;
}

// This is more legacy from PGBAR I think, but could be useful at some later point.
double GCaMP::getAmplitude(){
    Scope diagnostic("GCaMP.getAmplitude");
    diagnostic.mode(TSMode);
    init();

    std::array<int,100> spikes{};
    spikes[20]=1;
    std::array<double,100> resp;

    for(unsigned int i=0;i<100;i++){
        evolve(1e-3,spikes[i]);
        resp[i]=DFF;
    }

    diagnostic.array(csp_model::HistoryScan{resp.data(),resp.size()}, "amplitude_response", output_value);
    return(csp_model::max100_compat(resp));
}

// For kokkos port
void GCaMP::read_params(GCaMP_params & params)
{

  for (int i = 0; i < 14; i++) 
    params.Gparams[i] = Gparams(i);

  // parameters that are allowed to vary
  params.G_tot = G_tot;
  params.gamma = gamma;
  params.DCaT = DCaT;
  params.Rf = Rf;
  
  // parameters that can be fixed
  // indicator
  params.konN = konN;
  params.koffN = koffN;
  params.konC = konC;
  params.koffC = koffC;
  params.konPN = konPN;
  params.koffPN = koffPN;
  params.konPN2 = konPN2;
  params.koffPN2 = koffPN2;
  params.konPC = konPC;
  params.koffPC = koffPC;
  params.konPC2 = konPC2;
  params.koffPC2 = koffPC2;

  //calcium
  params.sigma2_calcium_spike = sigma2_calcium_spike;
  params.gam_in = gam_in;
  params.gam_out = gam_out;

  // buffer
  params.BCa0 = BCa0;
  params.kapB = kapB;
  
  // dFF normalization
  params.G0 = G0;
  params.Gsat = Gsat;
  params.Ginit = Ginit;

}