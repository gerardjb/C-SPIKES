#ifndef UTILS_H
#define UTILS_H
#include<armadillo>
#include <gsl/gsl_randist.h>

using namespace std;

namespace utils {
    void w_from_logW(const double*,double*,unsigned int);
    double Z_from_logW(const double *,unsigned int n);
    double Z_factor(double,double,double,double);



    
    template <int N>
    int discrete_from_uniform_random(const double *probs, double u_noise){
        
        // We have uniform noise between 0 and 1, so we can use this to sample the 
        // discrete distribution

        // Normalize the probabilities
        double totalP = 0.0;
        double probs_norm[N];
        for(int k=0; k<N; k++)
            totalP += probs[k];
        for(int k=0; k<N; k++)
            probs_norm[k] = probs[k]/totalP;

        double C[N+1];
        C[0] = 0.0;
        for(int k=0; k<N; k++)
            C[k+1] = C[k] + probs_norm[k];
        
        int idx;
        for(idx=0; idx<(N+1); idx++)
            if(C[idx] <= u_noise && u_noise < C[idx+1])
                break;

        return idx;
    }

    size_t gsl_ran_discrete_from_uniform(const gsl_ran_discrete_t *g, double u);
}



namespace stats{
    double gsl_gamma_logpdf(const double, const double,const double);
}

#endif
