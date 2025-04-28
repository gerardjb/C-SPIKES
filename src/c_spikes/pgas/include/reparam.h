#ifndef REPARAM_H
#define REPARAM_H
#include <gsl/gsl_spline.h>

class reparam {
public:
    reparam(const int N);
    ~reparam();
    void map(double, double, double, double&, double&, double&);

private:
    gsl_interp_accel *acc;
    double taur_over_taud(double x);

    gsl_spline *spline;
    gsl_interp *poly;
    double *x;
    double *y;
};

#endif
