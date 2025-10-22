#include "include/constants.h"
#include <cmath>
#include <json/json.h>
#include <fstream>
#include <iostream>

using namespace std;

constpar::constpar(string filename)
{
    Json::Reader reader;
    Json::Value  cfg;
    ifstream     paramfile(filename);

    paramfile >> cfg;

    // Integrated Brownian motion 
    bm_sigma  = cfg["BM"]["bm_sigma"].asDouble();   

    // data
    sampling_frequency = cfg["data"]["sampling_frequency"].asDouble();

    // Preprocessing
    baseline_frames     = cfg["preproc"]["baseline_frames"].asInt();
    nospike_before      = cfg["preproc"]["nospike_before"].asInt();
    normalization_index = cfg["preproc"]["normalization_index"].asInt();

    // Priors
    alpha_rate_b0        = cfg["priors"]["alpha rate 0"].asDouble();
    beta_rate_b0         = cfg["priors"]["beta rate 0"].asDouble();
    alpha_rate_b1        = cfg["priors"]["alpha rate 1"].asDouble();
    beta_rate_b1         = cfg["priors"]["beta rate 1"].asDouble();
    alpha_w01       = cfg["priors"]["alpha w01"].asDouble();
    beta_w01        = cfg["priors"]["beta w01"].asDouble();
    alpha_w10       = cfg["priors"]["alpha w10"].asDouble();
    beta_w10        = cfg["priors"]["beta w10"].asDouble();
    alpha_sigma2    = cfg["priors"]["alpha sigma2"].asDouble();
    beta_sigma2     = cfg["priors"]["beta sigma2"].asDouble();
    G_tot_mean      = cfg["priors"]["G_tot mean"].asDouble();
    G_tot_sd        = cfg["priors"]["G_tot sd"].asDouble();
    gamma_mean      = cfg["priors"]["gamma mean"].asDouble();
    gamma_sd        = cfg["priors"]["gamma sd"].asDouble();
    DCaT_mean       = cfg["priors"]["DCaT mean"].asDouble();
    DCaT_sd         = cfg["priors"]["DCaT sd"].asDouble();
    Rf_mean         = cfg["priors"]["Rf mean"].asDouble();
    Rf_sd           = cfg["priors"]["Rf sd"].asDouble();
    gam_in_mean     = cfg["priors"]["gam_in mean"].asDouble();
    gam_in_sd       = cfg["priors"]["gam_in sd"].asDouble();
    gam_out_mean    = cfg["priors"]["gam_out mean"].asDouble();
    gam_out_sd      = cfg["priors"]["gam_out sd"].asDouble();

    seed                = cfg["MCMC"]["seed"].asInt();
    niter               = cfg["MCMC"]["niter"].asInt();
    nparticles          = cfg["MCMC"]["nparticles"].asInt();
    SAMPLE_PARAMETERS   = cfg["MCMC"]["SAMPLE_PARAMETERS"].asBool();
    KNOWN_SPIKES        = cfg["MCMC"]["KNOWN_SPIKES"].asBool();
    maxspikes          = cfg["MCMC"]["maxspikes"].asInt();

    // proposals
    G_tot_prop_sd   = cfg["proposals"]["G_tot prop sd"].asDouble();
    gamma_prop_sd  = cfg["proposals"]["gamma prop sd"].asDouble();
    DCaT_prop_sd   = cfg["proposals"]["DCaT prop sd"].asDouble();
    Rf_prop_sd     = cfg["proposals"]["Rf prop sd"].asDouble();
    gam_in_prop_sd  = cfg["proposals"]["gam_in prop sd"].asDouble();
    gam_out_prop_sd = cfg["proposals"]["gam_out prop sd"].asDouble();

    // Integration of the GCaMP model
    TSMode = cfg["ODE"]["time step mode"].asInt();

}

void constpar::set_time_scales()
{
}

void constpar::print()
{
    cout << "________SETTINGS_____________" << endl;
    cout << "_____________________________" << endl;
    cout << "Brownian motion" << endl;
    cout << "    sigma = " << bm_sigma << endl;
    cout << "    sampling frequency = " << sampling_frequency << endl;
    cout << "priors" << endl;
    cout << "    alpha_sigma2 = " << alpha_sigma2 << endl;
    cout << "    beta_sigma2 = " << beta_sigma2 << endl;
    cout << "MCMC" << endl;
    cout << "    niter = " << niter << endl;
    cout << "    SAMPLE_PARAMETERS = " << SAMPLE_PARAMETERS << endl;
    cout << "    CROP_TRACE          = " << CROP_TRACE << endl;
    cout << "_____________________________" << endl;
}
