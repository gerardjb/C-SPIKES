#ifndef PARAM_H
#define PARAM_H

#include<fstream>
#include "constants.h"
#include "reparam.h"

class param {
public:
    double G_tot;
    double gamma;
    double DCaT;
    double Rf;
    double gam_in;
    double gam_out;
    double ca_half;
    double n_gate;
    double r0, r1;
    double* wbb;
    double sigma2;
    void print();
    void write(ofstream&,double);
    param& operator=(const param&);
    string filename;
    param();
    param(string);
    double logPrior(const constpar&);

};

#endif
