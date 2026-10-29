#pragma once
#include "particle.h"
#include "test_access.h"
#include <sstream>
#include <vector>
#include <iomanip>
struct Bytes {
 std::string s;
 template<class T> void scalar(const T& x){s.append(reinterpret_cast<const char*>(&x),sizeof(x));}
 template<class T> void array(const T& x){scalar(x.n_rows);scalar(x.n_cols);if(x.n_elem)s.append(reinterpret_cast<const char*>(x.memptr()),x.n_elem*sizeof(*x.memptr()));}
 void text(const std::string& x){scalar(x.size());s+=x;}
};
inline std::string config_bytes(const constpar& c){Bytes b;
 b.scalar(c.bm_sigma);
 b.scalar(c.alpha_sigma2);
 b.scalar(c.beta_sigma2);
 b.scalar(c.alpha_rate_b0);
 b.scalar(c.beta_rate_b0);
 b.scalar(c.alpha_rate_b1);
 b.scalar(c.beta_rate_b1);
 b.scalar(c.alpha_w01);
 b.scalar(c.beta_w01);
 b.scalar(c.alpha_w10);
 b.scalar(c.beta_w10);
 b.scalar(c.G_tot_mean);
 b.scalar(c.G_tot_sd);
 b.scalar(c.gamma_mean);
 b.scalar(c.gamma_sd);
 b.scalar(c.DCaT_mean);
 b.scalar(c.DCaT_sd);
 b.scalar(c.Rf_mean);
 b.scalar(c.Rf_sd);
 b.scalar(c.gam_in_mean);
 b.scalar(c.gam_in_sd);
 b.scalar(c.gam_out_mean);
 b.scalar(c.gam_out_sd);
 b.scalar(c.G_tot_prop_sd);
 b.scalar(c.gamma_prop_sd);
 b.scalar(c.DCaT_prop_sd);
 b.scalar(c.Rf_prop_sd);
 b.scalar(c.gam_in_prop_sd);
 b.scalar(c.gam_out_prop_sd);
 b.scalar(c.sampling_frequency);
 b.scalar(c.physics_frequency_hz);
 b.scalar(c.substeps_per_frame);
 b.scalar(c.min_substeps);
 b.scalar(c.time_integrated_observations);
 b.scalar(c.num_substeps);
 b.scalar(c.sim_dt);
 b.scalar(c.MOVE_SPIKES);
 b.scalar(c.SAMPLE_KINETICS);
 b.scalar(c.SAMPLE_SPIKES);
 b.scalar(c.SAMPLE_PARAMETERS);
 b.scalar(c.CROP_TRACE);
 b.scalar(c.KNOWN_SPIKES);
 b.scalar(c.c0_is_first_y);
 b.scalar(c.seed);
 b.scalar(c.niter);
 b.scalar(c.nparticles);
 b.scalar(c.maxspikes);
 b.scalar(c.t_min);
 b.scalar(c.t_max);
 b.scalar(c.baseline_frames);
 b.scalar(c.nospike_before);
 b.scalar(c.normalization_index);
 b.scalar(c.TSMode);
 b.text(c.output_folder);b.text(c.tag);return b.s;
}
inline std::string param_bytes(const param& p){Bytes b;for(double x:{p.G_tot,p.gamma,p.DCaT,p.Rf,p.gam_in,p.gam_out,p.r0,p.r1,p.sigma2,p.wbb[0],p.wbb[1]})b.scalar(x);b.text(p.filename);return b.s;}
inline std::string trajectory_bytes(const Trajectory& t){Bytes b;b.scalar(t.size);b.text(t.filename);b.array(t.B);b.array(t.burst);b.array(t.C);b.array(t.S);b.array(t.Y);return b.s;}
inline std::string public_caller(const SMC& m,const constpar& c,const param& p,const Trajectory& t){
 Bytes b;b.scalar(m.nparticles);b.scalar(m.TIME);b.array(m.data_time);b.array(m.data_y);
 for(auto row:m.particleSystem)for(unsigned j=0;j<m.nparticles;++j){const auto& x=row[j];b.array(x.C);b.scalar(x.B);b.scalar(x.burst);b.scalar(x.S);b.scalar(x.logWeight);b.scalar(x.ancestor);b.scalar(x.index);}
 b.text(config_bytes(c));b.text(param_bytes(p));b.text(trajectory_bytes(t));return b.s;
}
inline std::vector<double> handoff_values(const GCaMP_params& p){
 std::vector<double> v(p.Gparams,p.Gparams+14);
 for(double x:{p.G_tot,p.gamma,p.DCaT,p.Rf,p.konN,p.koffN,p.konC,p.koffC,p.konPN,p.koffPN,p.konPN2,p.koffPN2,p.konPC,p.koffPC,p.konPC2,p.koffPC2,p.c0,p.FWHM,p.sigma2_calcium_spike,p.gam_in,p.gam_out,p.koff_B,p.kon_B,p.B_tot,p.BCa0,p.kapB,p.csat,p.G0,p.Gsat,p.Ginit})v.push_back(x);return v;
}
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
namespace checked {
struct SweepAccess {
 static void seed(SMC& m){m.verbose=false;m.ancestor_sweep=29;}
 static std::string private_state(const SMC& m){Bytes b;b.scalar(m.ancestor_sweep);b.scalar(m.verbose);b.scalar(m.resampling.device);b.scalar(m.resampling.selected_trajectory);b.scalar(m.resampling.seed);b.array(m.tracemat);b.s.append(static_cast<const char*>(gsl_rng_state(m.rng)),gsl_rng_size(m.rng));return b.s;}
 static std::vector<const void*> addresses(const SMC& m){std::vector<const void*> a={&m,m.model,m.constants,m.rng,gsl_rng_state(m.rng),m.data_time.memptr(),m.data_y.memptr()};for(auto row:m.particleSystem){a.push_back(row);for(unsigned j=0;j<m.nparticles;++j)a.push_back(row[j].C.memptr());}return a;}
};
}
#endif
