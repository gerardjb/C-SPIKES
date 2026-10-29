#pragma once
#include "GCaMP_model.h"
#include "history_arma_bridge.h"
#include <string>
#include <vector>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <initializer_list>
namespace checked {
struct TestAccess {
 static void norms_for_test(GCaMP& m,double a,double b,double c,double rf){m.G0=a;m.Gsat=b;m.Ginit=c;m.Rf=rf;}

 static std::array<double,3> norms(const GCaMP& m){return {m.G0,m.Gsat,m.Ginit};}
 static double buffer_initial(const GCaMP& m){return m.BCa0;}
 static std::string without_cache(const GCaMP& src){GCaMP m(src);m.konN=0.;m.konC=0.;m.Gmat.fill(0.);return snapshot(m);}

 static std::array<double,6> cell_scalars(const GCaMP& m){return {m.G_tot,m.gamma,m.DCaT,m.Rf,m.gam_in,m.gam_out};}

 static std::array<double,2> rates(const GCaMP& m){return {m.konN,m.konC};}
 static void invalidate_normalization(GCaMP& m){m.G0=0.;m.Gsat=0.;m.Ginit=0.;m.Rf=2.;}

 static std::size_t rows(const csp_model::HistoryMatrix& m){return m.rows();}
 static std::size_t cols(const csp_model::HistoryMatrix& m){return m.cols();}
 static const double* data(const csp_model::HistoryMatrix& m){return m.data();}
using Vector=csp_model::HistoryVector;
 static std::size_t vsize(const arma::vec& v){return v.n_elem;}
 static std::size_t vrows(const arma::vec& v){return v.n_rows;}
 static std::size_t vcols(const arma::vec& v){return v.n_cols;}
 static const double* vdata(const arma::vec& v){return v.memptr();}
 static std::size_t vsize(const csp_model::HistoryVector& v){return v.size();}
 static std::size_t vrows(const csp_model::HistoryVector& v){return v.size();}
 static std::size_t vcols(const csp_model::HistoryVector&){return 1;}
 static const double* vdata(const csp_model::HistoryVector& v){return v.data();}
 template<class V>static void fill_vector(V& v,std::initializer_list<double> values){v.set_size(values.size());std::size_t i=0;for(double x:values)v(i++)=x;}
 template<class V>static std::string vector_bytes(const V& v){std::string s;arma::uword r=vrows(v),c=vcols(v);s.append(reinterpret_cast<const char*>(&r),sizeof(r));s.append(reinterpret_cast<const char*>(&c),sizeof(c));if(vsize(v))s.append(reinterpret_cast<const char*>(vdata(v)),vsize(v)*sizeof(double));return s;}
 static std::size_t dff_size(const GCaMP& m){return vsize(m.DFF_values);}
 static auto defined_bridge(GCaMP& m){
  arma::vec x={0.,.0001,.0002,.0003},y={1.,-2.,0.,4.},t={0.,.00005,.0002,.0003,.0004};
  m.DFF_values.reset();arma::vec output;arma::interp1(x,y,t,output,"linear");csp_model::copy_from_arma(m.DFF_values,output);
  return m.getDFFValues();
 }
 static std::string container_contract(){Vector a;a.set_size(3);for(unsigned i=0;i<3;++i)a(i)=i+.25;Vector b(a);if(vdata(a)==vdata(b))throw std::runtime_error("copy alias");b=b;Vector c(std::move(b));a(0)=-9;if(c(0)!=.25)throw std::runtime_error("move independence");b=std::move(c);b.reset();if(vsize(b)!=0)throw std::runtime_error("reset");b.set_size(6);for(unsigned i=0;i<6;++i)b(i)=10+i;return vector_bytes(b);}

 template<class M>static std::string matrix_bytes(const M& m){std::string s;arma::uword r=rows(m),c=cols(m);s.append(reinterpret_cast<const char*>(&r),sizeof(r));s.append(reinterpret_cast<const char*>(&c),sizeof(c));if(r*c)s.append(reinterpret_cast<const char*>(data(m)),r*c*sizeof(double));return s;}
 template<class M>static void fill_matrix(M& m,std::size_t r,std::size_t c,double v){m.set_size(r,c);for(std::size_t j=0;j<c;++j)for(std::size_t i=0;i<r;++i)m(i,j)=v;}
 static arma::mat fine(const GCaMP& m){arma::mat x(rows(m.G_values),cols(m.G_values));if(x.n_elem)std::copy(data(m.G_values),data(m.G_values)+x.n_elem,x.begin());return x;}
 static void pattern(GCaMP& m){fill_vector(m.DFF_values,{1.25,-2.25,0.,4.25,9.25});fill_matrix(m.G_interp,2,9,0.);for(std::size_t c=0;c<9;++c)for(std::size_t r=0;r<2;++r)m.G_interp(r,c)=r+10*c+.25;fill_vector(m.BCa_values,{3.25,4.25,5.25});fill_vector(m.Ca_values,{6.25});fill_vector(m.Ca_in_values,{});}
 static void edit_copy(GCaMP& m){m.DFF_values(0)=-111.;}
 static void self_assign(GCaMP& m){m.DFF_values=m.DFF_values;}
 static std::vector<const void*> history_addresses(const GCaMP& m){return {data(m.G_values),data(m.G_interp),vdata(m.DFF_values),vdata(m.BCa_values),vdata(m.Ca_values),vdata(m.Ca_in_values)};}
 static std::string non_dff_histories(const GCaMP& m){return matrix_bytes(m.G_values)+matrix_bytes(m.G_interp)+vector_bytes(m.BCa_values)+vector_bytes(m.Ca_values)+vector_bytes(m.Ca_in_values);}

 static std::string snapshot(const GCaMP& m) {
  std::string s;
  auto scalar=[&](const auto& x){s.append(reinterpret_cast<const char*>(&x),sizeof(x));};
  auto array=[&](const auto& x){scalar(x.n_rows);scalar(x.n_cols);if(x.n_elem)s.append(reinterpret_cast<const char*>(x.memptr()),x.n_elem*sizeof(*x.memptr()));};
  auto state_array=[&](const auto& x){const arma::uword rows=12,cols=1;scalar(rows);scalar(cols);s.append(reinterpret_cast<const char*>(x.data()),12*sizeof(double));};
  auto parameter_array=[&](const auto& x){const arma::uword rows=x.n_elem,cols=1;scalar(rows);scalar(cols);s.append(reinterpret_cast<const char*>(x.data()),x.n_elem*sizeof(double));};
  state_array(m.state);state_array(m.initial_state);scalar(m.DFF);parameter_array(m.Gparams);parameter_array(m.Cparams);{const arma::uword rows=9,cols=9;scalar(rows);scalar(cols);s.append(reinterpret_cast<const char*>(m.Gmat.data()),81*sizeof(double));}
{const arma::uword rows=5,cols=1;scalar(rows);scalar(cols);s.append(reinterpret_cast<const char*>(m.brightStates.data()),5*sizeof(std::uint64_t));}
scalar(m.step_count);scalar(m.TSMode);
  for(double x:{m.G_tot,m.gamma,m.DCaT,m.Rf,m.gam_in,m.gam_out,m.konN,m.koffN,m.konC,m.koffC,m.konPN,m.koffPN,m.konPN2,m.koffPN2,m.konPC,m.koffPC,m.konPC2,m.koffPC2,m.c0,m.FWHM,m.sigma2_calcium_spike,m.koff_B,m.kon_B,m.B_tot,m.BCa0,m.kapB,m.csat,m.G0,m.Gsat,m.Ginit})scalar(x);
  s+=vector_bytes(m.DFF_values);s+=matrix_bytes(m.G_values);s+=matrix_bytes(m.G_interp);s+=vector_bytes(m.BCa_values);s+=vector_bytes(m.Ca_values);s+=vector_bytes(m.Ca_in_values);return s;
 }
 static std::string nonnumeric(const GCaMP& src){GCaMP m(src);m.state.fill(0.);m.initial_state.fill(0.);m.G0=m.Gsat=m.Ginit=0;return snapshot(m);}
 static void seed(GCaMP& m){fill_vector(m.DFF_values,{1.,2.});fill_matrix(m.G_values,2,9,1.);fill_matrix(m.G_interp,2,9,2.);fill_vector(m.BCa_values,{3.,4.});fill_vector(m.Ca_values,{5.,6.});fill_vector(m.Ca_in_values,{7.,8.});if(std::getenv("G2O_PATTERN"))pattern(m);if(std::getenv("G2O_SEED_NAN")){m.DFF_values(1)=std::numeric_limits<double>::quiet_NaN();m.Ca_values(0)=std::numeric_limits<double>::quiet_NaN();}}
 static std::string histories(const GCaMP& m){std::string s;auto a=[&](const auto& x){s.append(reinterpret_cast<const char*>(&x.n_rows),sizeof(x.n_rows));s.append(reinterpret_cast<const char*>(&x.n_cols),sizeof(x.n_cols));if(x.n_elem)s.append(reinterpret_cast<const char*>(x.memptr()),x.n_elem*sizeof(*x.memptr()));};s+=vector_bytes(m.DFF_values);s+=matrix_bytes(m.G_values);s+=matrix_bytes(m.G_interp);s+=vector_bytes(m.BCa_values);s+=vector_bytes(m.Ca_values);s+=vector_bytes(m.Ca_in_values);return s;}
 static std::vector<const void*> addresses(const GCaMP& m){return {&m,m.state.data(),m.initial_state.data(),m.Gmat.data(),m.Gparams.data(),m.Cparams.data(),vdata(m.DFF_values),data(m.G_values),data(m.G_interp),vdata(m.BCa_values),vdata(m.Ca_values),vdata(m.Ca_in_values)};}

};
}
