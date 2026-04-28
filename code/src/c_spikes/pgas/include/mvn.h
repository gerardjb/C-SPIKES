#ifndef MVN_H 
#define MVN_H 

#include<armadillo>
#include<gsl/gsl_rng.h>

class mvn {

public:

    // Constructors
    mvn();
    mvn(gsl_rng* r);
    mvn(gsl_rng* r, const arma::vec &mu, const arma::mat &S);

    //~mvn();

    void setRNG(gsl_rng *r);
    void setParams(const arma::vec &mu,const arma::mat &S);
    void printL();

    // Multivariate Normal
    arma::mat rmvn_mat(int N);
    arma::vec rmvn_vec();
    arma::vec rmvn_vec(const arma::vec &mu,const arma::mat &S);
    arma::mat rmvn_mat(const arma::vec &mu,const arma::mat &S, int N);
    double dmvn(const arma::vec &x);
    double dmvn(const arma::vec &x, const arma::vec &mu, const arma::mat S);
    double dmvn_log(const arma::vec &x);
    double dmvn_log(const arma::vec &x, const arma::vec &mu, const arma::mat S);
    arma::mat getInvCov();


    // Wishart
    arma::mat rWishart(int v);
    arma::mat rWishart(int v, const arma::mat &scale);
    arma::mat rInvWishart(int v);
    arma::mat rInvWishart(int v, const arma::mat &scale);

private:
    gsl_rng* rng;
    arma::mat Cov;
    arma::mat invCov;
    arma::mat L;
    arma::vec mean;
    int size;
};

#endif
