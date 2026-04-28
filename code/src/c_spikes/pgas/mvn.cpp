#include "include/mvn.h"
#include "gsl/gsl_randist.h"

mvn::mvn()
{
    rng = 0;
}

mvn::mvn(gsl_rng *ran)
{
    rng = ran;
}

mvn::mvn(gsl_rng *ran, const arma::vec&mu, const arma::mat&S)
{
    rng    = ran;
    mean   = mu;
    Cov    = S;
    invCov = inv(Cov);
    size   = mu.n_rows;
    L      = arma::chol(S, "lower");
}

// This was stupid... the mvn class never allocate the rng so there is no need to delete it
// as it is created somewhere else

/*mvn::~mvn(){
 *  if(rng!=NULL) gsl_rng_free(rng);
 *  std::cout<<"rng deleted!"<<std::endl;
 * }
 */

void mvn::setRNG(gsl_rng *r)
{
    rng = r;
}

void mvn::setParams(const arma::vec&mu, const arma::mat&S)
{
    mean   = mu;
    Cov    = S;
    invCov = inv(Cov);
    size   = S.n_rows;
    L      = arma::chol(S, "lower");
}

void mvn::printL()
{
    std::cout << "L" << std::endl;
    L.print();
    std::cout << "L*Lt" << std::endl;
    (L * L.t()).print();
    std::cout << "Cov" << std::endl;
    Cov.print();
}

arma::vec mvn::rmvn_vec()
{
    arma::vec z(size);

    for (unsigned int i = 0; i < size; ++i)
    {
        z(i) = gsl_ran_gaussian(rng, 1.0);
    }
    return(mean + L * z);
}

arma::mat mvn::rmvn_mat(int N)
{
    arma::mat    Z(size, N);
    unsigned int i, j;

    for (j = 0; j < N; ++j)
    {
        for (i = 0; i < size; ++i)
        {
            Z(i, j) = gsl_ran_gaussian(rng, 1.0);
        }
    }

    arma::mat LZ = L * Z;

    return(LZ.each_col() + mean);
}

arma::vec mvn::rmvn_vec(const arma::vec&mu, const arma::mat&S)
{
    setParams(mu, S);
    return(rmvn_vec());
}

arma::mat mvn::rmvn_mat(const arma::vec&mu, const arma::mat&S, int N)
{
    setParams(mu, S);
    return(rmvn_mat(N));
}

double mvn::dmvn(const arma::vec&x)
{
    double cc = 1. / pow(2 * arma::datum::pi, size / 2.) / sqrt(arma::det(Cov));
    double qf = arma::as_scalar((x - mean).t() * invCov * (x - mean));

    return(cc * exp(-0.5 * qf));
}

double mvn::dmvn_log(const arma::vec&x)
{
    double logp = 0;

    logp += -size / 2. * log(2 * arma::datum::pi);
    logp += -0.5 * log(det(Cov));
    logp += -0.5 * arma::as_scalar((x - mean).t() * invCov * (x - mean));

    return(logp);
}

double mvn::dmvn(const arma::vec&x, const arma::vec&mu, const arma::mat S)
{
    arma::mat invS = inv(S);
    int       dim  = mu.n_rows;
    double    cc   = 1. / pow(2 * arma::datum::pi, dim / 2.) / sqrt(arma::det(S));
    double    qf   = arma::as_scalar((x - mu).t() * invS * (x - mu));

    return(cc * exp(-0.5 * qf));
}

double mvn::dmvn_log(const arma::vec&x, const arma::vec&mu, const arma::mat S)
{
    double    logp = 0;
    arma::mat invS = inv(S);
    int       dim  = mu.n_rows;

    logp += -0.5 * dim * log(2 * arma::datum::pi);
    logp += -0.5 * log(det(S));
    logp += -0.5 * arma::as_scalar((x - mu).t() * invS * (x - mu));

    return(logp);
}

arma::mat mvn::getInvCov()
{
    return(invCov);
}

arma::mat mvn::rWishart(int v)
{
    arma::mat    A(arma::size(Cov), arma::fill::zeros);
    unsigned int i, k;

    for (k = 0; k < size; ++k)
    {
        A.at(k, k) = sqrt(gsl_ran_chisq(rng, v - (k + 1) + 1));
    }
    for (k = 0; k < size - 1; ++k)
    {
        for (i = k + 1; i < size; ++i)
        {
            A(i, k) = gsl_ran_gaussian(rng, 1.0);
        }
    }
    arma::mat LA = L * A;

    return(LA * LA.t());
}

arma::mat mvn::rWishart(int v, const arma::mat&scale)
{
    setParams(arma::zeros <arma::vec>(scale.n_rows), scale);
    return(rWishart(v));
}

arma::mat mvn::rInvWishart(int v)
{
    arma::mat    A(arma::size(Cov), arma::fill::zeros);
    unsigned int i, k;

    for (k = 0; k < size; ++k)
    {
        A.at(k, k) = sqrt(gsl_ran_chisq(rng, v - (k + 1) + 1));
    }
    for (k = 0; k < size - 1; ++k)
    {
        for (i = k + 1; i < size; ++i)
        {
            A(i, k) = gsl_ran_gaussian(rng, 1.0);
        }
    }

    arma::mat invAt = inv(A).t();
    arma::mat LA    = L * invAt;

    return(LA * LA.t());
}

arma::mat mvn::rInvWishart(int v, const arma::mat&scale)
{
    setParams(arma::zeros <arma::vec>(scale.n_rows), scale);
    return(rInvWishart(v));
}

//inv(L*(Lt)) = inv(L)t * inv(L)

//inv(L)t AAt inv(L)
//L inv(AAt) Lt

// decomposing S = L Lt
// Zt Lt inv(Lt) inv(L) LZ = Zt Z
