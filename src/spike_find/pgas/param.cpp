#include "include/param.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include<cmath>

using namespace std;

param::param(){
		G_tot = 0.0;
		gamma = 0.0;
		DCaT = 0.0;
		Rf = 0.0;
		gam_in = 0.0;
		gam_out = 0.0;
		sigma2 = 0.0;
		r0 = 0.0;
		r1 = 0.0;
    wbb = new double[2];
}

param::param(string fname){
	G_tot = 0.0;
		gamma = 0.0;
		DCaT = 0.0;
		Rf = 0.0;
		gam_in = 0.0;
		gam_out = 0.0;
		sigma2 = 0.0;
		r0 = 0.0;
		r1 = 0.0;
    wbb = new double[2];
    filename=fname;
}

void param::print(){
    cout<<"G_tot  = "<<G_tot<<endl
        <<"gamma  = "<<gamma<<endl
        <<"DCaT   = "<<DCaT<<endl
        <<"Rf     = "<<Rf<<endl
        <<"gam_in = "<<gam_in<<endl
        <<"gam_out = "<<gam_out<<endl
        <<"sigma2 = "<<sigma2<<endl
        <<"r0     = "<<r0<<endl
        <<"r1     = "<<r1<<endl;
}

void param::write(ofstream &out,double sf){
    if(!out.is_open()){
        out.open(filename);
        out<<"G_tot,gamma,DCaT,Rf,gam_in,gam_out,sigma2,r0,r1,w01,w10"<<endl;
    }

    out<<G_tot<<","
        <<gamma<<","
        <<DCaT<<","
        <<Rf<<","
        <<gam_in<<","
        <<gam_out<<","
        <<sigma2<<","
        <<r0<<","
        <<r1<<","
        <<wbb[0]<<","
        <<wbb[1]<<endl;
}

param& param::operator=(const param& p){
    G_tot    = p.G_tot;
    gamma    = p.gamma;
    DCaT     = p.DCaT;
    Rf       = p.Rf;
    gam_in   = p.gam_in;
    gam_out  = p.gam_out;
    r0       = p.r0;
    r1       = p.r1;
    wbb[0]   = p.wbb[0];
    wbb[1]   = p.wbb[1];
    sigma2   = p.sigma2;
    return *this;
}

double param::logPrior(const constpar& constants){
    
    double logp=0;

    logp += -0.5*pow((G_tot  - constants.G_tot_mean) / constants.G_tot_sd,2)  
            -0.5*pow((gamma  - constants.gamma_mean) / constants.gamma_sd,2) 
            -0.5*pow((DCaT   - constants.DCaT_mean)  / constants.DCaT_sd,2) 
            -0.5*pow((Rf     - constants.Rf_mean)    / constants.Rf_sd,2)
            -0.5*pow((gam_in - constants.gam_in_mean)/constants.gam_in_sd,2)
            -0.5*pow((gam_out - constants.gam_out_mean)/constants.gam_out_sd,2);

    return logp;
}

