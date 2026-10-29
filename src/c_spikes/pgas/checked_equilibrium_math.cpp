#ifdef C_SPIKES_CHECKED_CPU
#include "include/checked_equilibrium.h"
#include <gsl/gsl_linalg.h>
#include <algorithm>
#include <cmath>
namespace checked { namespace detail {
namespace {
constexpr double eps=std::numeric_limits<double>::epsilon();
template<class A> bool finite(const A& a){for(double x:a)if(!std::isfinite(x))return false;return true;}
double at(const Matrix9& a,int i,int j){return a[9*i+j];}
double infnorm(const Matrix9& a){double v=0;for(int i=0;i<9;++i){double s=0;for(int j=0;j<9;++j)s+=std::abs(at(a,i,j));v=std::max(v,s);}return v;}
bool valid_input(const Matrix9& q,double mass){return finite(q)&&std::isfinite(mass)&&mass>0;}
}
Factors gsl_backend(const Matrix9& q){
 Factors f;f.u=q;Vector9 workspace{};
 auto u=gsl_matrix_view_array(f.u.data(),9,9);auto v=gsl_matrix_view_array(f.v.data(),9,9);
 auto s=gsl_vector_view_array(f.s.data(),9);auto work=gsl_vector_view_array(workspace.data(),9);
 f.code=gsl_linalg_SV_decomp(&u.matrix,&v.matrix,&s.vector,&work.vector);return f;
}
std::array<int,3> classify(const Vector9& s){
 std::array<int,3> ranks{};const double tau=9*eps*s[0],factors[3]={.1,1,10};
 for(double x:s)for(int k=0;k<3;++k)ranks[k]+=x>tau*factors[k];return ranks;
}
bool normalize(const Vector9& z,double mass,Vector9& output,double& sum,double& guard){
 sum=0;double l1=0;for(double x:z){sum+=x;l1+=std::abs(x);}guard=64*eps*l1;
 if(!finite(z)||!std::isfinite(mass)||mass<=0||!std::isfinite(sum)||!std::isfinite(guard)||!(std::abs(sum)>guard))return false;
 Vector9 g{};for(int i=0;i<9;++i)g[i]=z[i]/sum*mass;
 if(!finite(g))return false;output=g;return true;
}
bool positive(const Vector9& g,double mass){
 if(!std::isfinite(mass)||mass<=0||!finite(g))return false;
 for(double x:g)if(x/mass < -1e-12)return false;return true;
}
Result assess(const Matrix9& q,const Factors& f,bool generator,double mass){
 Result r;if(!valid_input(q,mass))return r;
 r.backend_called=true;r.factors=f;
 if(f.code!=0){r.outcome=Outcome::backend_failed;return r;}
 r.sorted=finite(f.s)&&finite(f.u)&&finite(f.v);
 for(int i=0;i<9;++i)r.sorted=r.sorted&&f.s[i]>=0&&(i==0||f.s[i]<=f.s[i-1]);
 if(!r.sorted){r.outcome=Outcome::malformed_factors;return r;}
 r.tau=9*eps*f.s[0];r.ranks=classify(f.s);r.nullity=9-r.ranks[1];
 Matrix9 reconstruction{},ou{},ov{};
 for(int i=0;i<9;++i)for(int j=0;j<9;++j){
  double x=0,uu=0,vv=0,pp=0;
  for(int k=0;k<9;++k){x+=at(f.u,i,k)*f.s[k]*at(f.v,j,k);uu+=at(f.u,k,i)*at(f.u,k,j);vv+=at(f.v,k,i)*at(f.v,k,j);if(f.s[k]<=r.tau)pp+=at(f.v,i,k)*at(f.v,j,k);}
  reconstruction[9*i+j]=x-at(q,i,j);ou[9*i+j]=uu-(i==j);ov[9*i+j]=vv-(i==j);r.projector[9*i+j]=pp;
 }
 const double qnorm=infnorm(q),error=infnorm(reconstruction);
 r.reconstruction=qnorm>0?error/qnorm:error;r.u_orthogonality=infnorm(ou);r.v_orthogonality=infnorm(ov);r.has_diagnostics=true;
 if(!finite(reconstruction)||!finite(ou)||!finite(ov)||!finite(r.projector)||!std::isfinite(qnorm)||!std::isfinite(r.reconstruction)||!std::isfinite(r.v_orthogonality)||r.reconstruction>1e-12||r.v_orthogonality>1e-12){r.outcome=Outcome::svd_quality_failed;return r;}
 if(r.nullity==0){r.outcome=Outcome::no_nullspace;return r;}
 if(r.nullity>1){r.outcome=Outcome::nonunique;return r;}
 if(r.ranks!=std::array<int,3>{{8,8,8}}){r.outcome=Outcome::rank_ambiguous;return r;}
 int k=0;while(k<9&&f.s[k]>r.tau)++k;
 if(k==9){r.outcome=Outcome::malformed_factors;return r;}
 Vector9 z{},candidate{};for(int i=0;i<9;++i)z[i]=at(f.v,i,k);
 if(!normalize(z,mass,candidate,r.sum,r.guard)){r.outcome=Outcome::normalization_unsafe;return r;}
 if(!generator){r.outcome=Outcome::analytic_no_selection;return r;}
 double total=0,minval=std::numeric_limits<double>::infinity(),maxabs=0,resid=0;
 for(double x:candidate){total+=x;minval=std::min(minval,x);maxabs=std::max(maxabs,std::abs(x));}
 for(int i=0;i<9;++i){double x=0;for(int j=0;j<9;++j)x+=at(q,i,j)*candidate[j];resid=std::max(resid,std::abs(x));}
 const double den=qnorm*maxabs;r.residual=den>0?resid/den:std::numeric_limits<double>::quiet_NaN();
 r.mass_error=std::abs(total-mass)/mass;r.minimum=minval/mass;
 if(!std::isfinite(den)||!std::isfinite(r.residual)||!std::isfinite(r.mass_error)||r.residual>1e-10||r.mass_error>1e-10||!positive(candidate,mass)){r.outcome=Outcome::invariant_failed;return r;}
 r.state=candidate;r.has_state=true;r.outcome=Outcome::candidate;return r;
}
Result evaluate(const Matrix9& q,bool generator,double mass,Backend backend){
 if(!valid_input(q,mass))return Result{};
 return assess(q,backend(q),generator,mass);
}
const char* outcome_name(Outcome x){
 switch(x){
 #define NAME(n) case Outcome::n:return #n
 NAME(input_invalid);NAME(backend_failed);NAME(malformed_factors);NAME(svd_quality_failed);NAME(no_nullspace);NAME(nonunique);NAME(rank_ambiguous);NAME(normalization_unsafe);NAME(analytic_no_selection);NAME(invariant_failed);NAME(candidate);
 #undef NAME
 }return "unknown";
}
const char* normalization_name(Outcome x){
 switch(x){case Outcome::candidate:return "normalized_unique";case Outcome::nonunique:return "nonunique_no_selection";case Outcome::rank_ambiguous:return "rank_ambiguous_no_selection";case Outcome::normalization_unsafe:return "normalization_sum_unsafe";case Outcome::analytic_no_selection:return "analytic_no_physical_selection";default:return outcome_name(x);}
}
}

}
#endif
