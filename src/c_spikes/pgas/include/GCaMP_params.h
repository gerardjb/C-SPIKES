#ifndef CSP_GCAMP_PARAMS_H
#define CSP_GCAMP_PARAMS_H

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

#endif
