#include<algorithm>
#include<gsl/gsl_math.h>
#include<armadillo>
#include"include/utils.h"
#include<gsl/gsl_sf.h>
#include <gsl/gsl_randist.h>
using namespace std;

namespace utils {

    void w_from_logW(const double *src, double* out, unsigned int n){
        double maxLogWeight = *max_element(src,src+n);
        for(unsigned int i=0;i<n;i++) out[i] = exp(src[i]-maxLogWeight);
    }

    double Z_from_logW(const double *src, unsigned int n){
        double maxLogWeight = *max_element(src,src+n);
        double Z=0;
        for(unsigned int i=0; i<n; i++){
            Z+=exp(src[i]-maxLogWeight);
        }
        return(Z*exp(maxLogWeight));
    }

    double Z_factor(double x,double xprop,double sigma,double cutoff){
        return(log(gsl_sf_erf_Q((cutoff-x)/sigma)/gsl_sf_erf_Q((cutoff-xprop)/sigma)));
    }

    // A helper function that generates a discrete random variable from a uniform sample.Add commentMore actions
    // This function lets us pre-generate noise samples for the move and weight steps
    // Ahead of time to aid in reproducibility. Implmented from gsl_ran_discrete().
    size_t gsl_ran_discrete_from_uniform(const gsl_ran_discrete_t *g, double u)
    {
        #define KNUTH_CONVENTION 1

        size_t c=0;
        double f;
    #if KNUTH_CONVENTION
        c = (u*(g->K));
    #else
        u *= g->K;
        c = u;
        u -= c;
    #endif
        f = (g->F)[c];
        /* fprintf(stderr,"c,f,u: %d %.4f %f\n",c,f,u); */
        if (f == 1.0) return c;

        if (u < f) {
            return c;
        }
        else {
            return (g->A)[c];
        }
    }

}

namespace stats {
    double gsl_gamma_logpdf(const double x, const double shape, const double scale){
        return ( (shape-1)*log(x) - x/scale );
    }
}

