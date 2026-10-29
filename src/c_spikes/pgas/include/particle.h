#ifndef PARTICLE_H
#define PARTICLE_H

#include<iostream>
#include"param.h"
#include"constants.h"
#include<armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include<string>
#include"GCaMP_model.h"
#include"particle_array.h"
#include"resampling_options.h"

#include <Kokkos_Core.hpp>

using namespace std;



class Particle
{
    public:
        Particle& operator=(const Particle&);
        double B;
        int burst;
        arma::vec C;
        int S;
        double logWeight;
        int ancestor;
        void print();
        Particle();
        int index;
};

class Trajectory
{
    public:
        Trajectory(unsigned int, string);
        unsigned int size;
        arma::vec B;
        arma::ivec burst;
        arma::vec C;  // this is only DFF
        arma::vec S;
        arma::vec Y;

        void write(ofstream &, unsigned int idx=0);
        double logp(const param*, const constpar*);
        Trajectory& operator=(const Trajectory&);

        string filename;

};

#ifdef C_SPIKES_CHECKED_SWEEP_CPU
namespace checked {
struct SweepPreparation {
    Report report;
    std::optional<GCaMP_params> handoff;
};
#ifdef C_SPIKES_CHECKED_SWEEP_RUN_CPU
struct SweepRun {
    Report preparation;
    bool sweep_returned = false;
};
#endif
struct SweepAccess;
}
#endif

class SMC{
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
    friend struct checked::SweepAccess;
#endif
    public:

        SMC(arma::vec,arma::vec,int,constpar&,bool,int seed=0, unsigned int maxlen=0, string GParam_file=""); 
        ~SMC();

        double logf(const Particle&,const Particle&, const param&);
        void move_and_weight(Particle&, const Particle&, double, const param &, double, double, bool); // move particles and weight
        void move_and_weight_GTS(Particle&, const Particle&, double, const param &, double, double, bool); // move particles and weight

        void rmu(Particle &, double, const param &, bool); // draw initial state
        
        void PGAS(const param &theta, const Trajectory &traj_in, Trajectory &traj_out);
#ifdef C_SPIKES_CHECKED_SWEEP_CPU
        checked::SweepPreparation try_prepare_sweep(const param&, const checked::Context&);
#endif

#ifdef C_SPIKES_CHECKED_SWEEP_RUN_CPU
        checked::SweepRun try_PGAS(const param&, const Trajectory&, Trajectory&, const checked::Context&);
#endif



        void sampleParameters(const param&, param &, Trajectory&);

        unsigned int nparticles;
        unsigned int TIME;

        arma::vec data_y;
        arma::vec data_time;

        vector<Particle*> particleSystem;

        GCaMP* model;

    private:
        void PGAS_impl(const param&, const Trajectory&, Trajectory&, bool prepared);
        pgas::ResamplingOptions resampling;
        uint64_t ancestor_sweep = 0;
        gsl_rng *rng;
        arma::mat tracemat;
        constpar *constants;
        bool verbose;

};

#endif
