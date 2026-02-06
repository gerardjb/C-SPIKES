#ifndef UTILS_H
#define UTILS_H
#include<armadillo>
#include <gsl/gsl_randist.h>

using namespace std;

namespace utils {
    void w_from_logW(const double*,double*,unsigned int);
    double Z_from_logW(const double *,unsigned int n);
    double Z_factor(double,double,double,double);

    template <class vectype1, class vectype2>
    void subsample(const vectype1 &x_in,vectype2 &x_out, double freq_in, double freq_out){
        arma::vec times_in = arma::linspace(0,x_in.n_elem/freq_in,x_in.n_elem);

        std::vector<double> tmp(1);
        tmp[0]=0;

        int counter=1;
        for(unsigned int i=0;i<x_in.n_elem;++i){
            while(times_in(i)>counter/freq_out){
                tmp.push_back(0);
                counter++;
            }
            if(x_in(i)>0) tmp[counter-1]+=1;
        }

        x_out = arma::vec(&tmp[0],counter);

    }

    template <class vectype1, class vectype2>
    double subsampled_and_filtered_correlation(const vectype1 &v1,const vectype2 &v2, double freq_in, double freq_out, double bw_ms){
        arma::vec ss1,ss2;

        subsample(v1,ss1,freq_in,freq_out);
        subsample(v2,ss2,freq_in,freq_out);

        if(ss1.n_elem!=ss2.n_elem) {
            cout<<"correlation between vectors of different sizes!! Exiting..."<<endl;
            exit(1);
        }

        arma::mat K = arma::zeros<arma::mat>(ss1.n_elem,ss1.n_elem);
        arma::mat::row_col_iterator it;
        double dt2=pow(1/freq_out,2);
        double sigma2=pow(bw_ms/1000.0,2);

        for (it=K.begin_row_col(); it!= K.end_row_col();++it)
            *it = pow(2*arma::datum::pi*sigma2,-0.5)*exp(-0.5/sigma2*dt2*pow((double)it.row()-(double)it.col(),2));
        

        arma::vec ss1_filtered, ss2_filtered;
        arma::vec z=arma::sum(K,1);

        K.each_col() /= z;

        ss1_filtered = K*ss1;
        ss2_filtered = K*ss2;

        return(arma::as_scalar(arma::cor(ss1_filtered,ss2_filtered)));
    }

    template <class vectype1, class vectype2>
    double subsampled_correlation(const vectype1 &v1, const vectype2 &v2, double freq_in, double freq_out){
        arma::vec ss1,ss2;
        subsample(v1,ss1,freq_in,freq_out);
        subsample(v2,ss2,freq_in,freq_out);
        return(arma::as_scalar(arma::cor(ss1,ss2)));
    }
    
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
