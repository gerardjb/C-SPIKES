#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
using namespace std;

class constpar {
public:
    constpar();
    constpar(string);

    void print();

    double bm_sigma = 1e-4;

    double alpha_sigma2;
    double beta_sigma2;      // Note that beta in GSL is actually the SCALE, not the rate.
    double alpha_rate_b0, beta_rate_b0;
    double alpha_rate_b1, beta_rate_b1;
    double alpha_w01, beta_w01;
    double alpha_w10, beta_w10;
    double G_tot_mean, G_tot_sd;
    double gamma_mean, gamma_sd;
    double DCaT_mean, DCaT_sd;
    double Rf_mean, Rf_sd;
    double gam_in_mean, gam_in_sd;
    double gam_out_mean, gam_out_sd;
    double G_tot_prop_sd;
    double gamma_prop_sd;
    double DCaT_prop_sd;
    double Rf_prop_sd;
    double gam_in_prop_sd;
    double gam_out_prop_sd;

    double sampling_frequency;

    bool MOVE_SPIKES         = true;
    bool SAMPLE_KINETICS     = true;
    bool SAMPLE_SPIKES       = true;
    bool SAMPLE_PARAMETERS   = true;
    bool CROP_TRACE          = false;
    bool KNOWN_SPIKES        = false;

    int seed;
    int niter;
    int nparticles;

    // Preprocessing
    int t_min, t_max;
    int baseline_frames;
    int nospike_before;
    int normalization_index;

    string output_folder;
    string tag;

    void set_time_scales();

    // Integration of the model
    int TSMode = 1;
};
#endif
