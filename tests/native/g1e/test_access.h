#pragma once
#include "GCaMP_model.h"
#include <string>
#include <vector>
namespace checked {
struct TestAccess {
 static std::string snapshot(const GCaMP& m) {
  std::string s;
  auto scalar=[&](const auto& x){s.append(reinterpret_cast<const char*>(&x),sizeof(x));};
  auto array=[&](const auto& x){scalar(x.n_rows);scalar(x.n_cols);if(x.n_elem)s.append(reinterpret_cast<const char*>(x.memptr()),x.n_elem*sizeof(*x.memptr()));};
  array(m.state);array(m.initial_state);scalar(m.DFF);array(m.Gparams);array(m.Cparams);array(m.Gmat);array(m.brightStates);scalar(m.step_count);scalar(m.TSMode);
  for(double x:{m.G_tot,m.gamma,m.DCaT,m.Rf,m.gam_in,m.gam_out,m.konN,m.koffN,m.konC,m.koffC,m.konPN,m.koffPN,m.konPN2,m.koffPN2,m.konPC,m.koffPC,m.konPC2,m.koffPC2,m.c0,m.FWHM,m.sigma2_calcium_spike,m.koff_B,m.kon_B,m.B_tot,m.BCa0,m.kapB,m.csat,m.G0,m.Gsat,m.Ginit})scalar(x);
  array(m.DFF_values);array(m.G_values);array(m.G_interp);array(m.BCa_values);array(m.Ca_values);array(m.Ca_in_values);return s;
 }
 static std::string nonnumeric(const GCaMP& src){GCaMP m(src);m.state.zeros();m.initial_state.zeros();m.G0=m.Gsat=m.Ginit=0;return snapshot(m);}
 static void seed(GCaMP& m){m.DFF_values={1.,2.};m.G_values=arma::ones<arma::mat>(2,9);m.G_interp=2*arma::ones<arma::mat>(2,9);m.BCa_values={3.,4.};m.Ca_values={5.,6.};m.Ca_in_values={7.,8.};}
 static std::string histories(const GCaMP& m){std::string s;auto a=[&](const auto& x){s.append(reinterpret_cast<const char*>(&x.n_rows),sizeof(x.n_rows));s.append(reinterpret_cast<const char*>(&x.n_cols),sizeof(x.n_cols));if(x.n_elem)s.append(reinterpret_cast<const char*>(x.memptr()),x.n_elem*sizeof(*x.memptr()));};a(m.DFF_values);a(m.G_values);a(m.G_interp);a(m.BCa_values);a(m.Ca_values);a(m.Ca_in_values);return s;}
 static std::vector<const void*> addresses(const GCaMP& m){return {&m,m.state.memptr(),m.initial_state.memptr(),m.Gmat.memptr(),m.Gparams.memptr(),m.Cparams.memptr(),m.DFF_values.memptr(),m.G_values.memptr(),m.G_interp.memptr(),m.BCa_values.memptr(),m.Ca_values.memptr(),m.Ca_in_values.memptr()};}

};
}
