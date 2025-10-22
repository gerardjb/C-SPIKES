#include"include/particle.h"
#include"include/particle_array.h"
#include"include/utils.h"
#include<gsl/gsl_sf_gamma.h>
#include<gsl/gsl_math.h>
#include<fstream>
#include<string>
#include<ctime>
#include"include/GCaMP_model.h"
#include<iomanip>


using namespace std;

// PARTICLE CLASS
// -----------------------------------------------------------
//
Particle::Particle(){
    C.set_size(12);
    B=0;
    burst=0;
    S=0;
    ancestor=-1;
    logWeight=0;
}

Particle& Particle::operator=(const Particle& p){
    B=p.B;
    burst=p.burst;
    C=p.C;
    S=p.S;
    logWeight=p.logWeight;
    return *this;
}

void Particle::print(){
    cout<<"    C    : "<<C(0)<<endl
        <<"    B    : "<<B<<endl
        <<"    burst: "<<burst<<endl
        <<"    S    : "<<S<<endl;
}

// Trajectory class
// ------------------------------------------------------------
//
Trajectory::Trajectory(unsigned int s, string fname){

    filename=fname;
    size=s;
    B.resize(s);
    burst.resize(s);
    C.resize(s);
    S.resize(s);
    S.zeros();
    Y.resize(s);

}

void Trajectory::simulate(gsl_rng* rng, const param& par, const constpar *constants){

    GCaMP model(par.G_tot, par.gamma, par.DCaT, par.Rf, par.gam_in, par.gam_out);

    double dt=1.0/constants->sampling_frequency;
    double rate[2] = {par.r0*dt,par.r1*dt};

    B(0) = 0;//gsl_ran_gaussian(rng,constants->bm_sigma);
    burst(0) = 0;
    S(0) = gsl_ran_poisson(rng,par.r0*dt);
    Y(0) = 0+B(0)+gsl_ran_gaussian(rng,sqrt(par.sigma2));
    
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};

    for(unsigned int t=1;t<size;t++){

        burst(t) = (gsl_rng_uniform(rng) < W[burst(t-1)][0]) ? 0 : 1;

        B(t)     = B(t-1)+gsl_ran_gaussian(rng,constants->bm_sigma*sqrt(dt)); 
        S(t)     = gsl_ran_poisson(rng,rate[burst(t)]);
        
        model.evolve(dt,S(t));

        C(t)     = model.DFF;
        Y(t)     = model.DFF+B(t)+gsl_ran_gaussian(rng,sqrt(par.sigma2)); 
    }
}

void Trajectory::write(ofstream &outfile, unsigned int index){
    
    if(!outfile.is_open()){
        outfile.open(filename);
        outfile<<"index,burst,B,S,C,Y"<<endl;
    }

    for(unsigned int i=0;i<size;++i){
        outfile<<index<<','
            <<burst(i)<<','
            <<B(i)<<','
            <<S(i)<<','
            <<C(i)<<','
            <<Y(i)<<endl;
    }

}

Trajectory& Trajectory::operator=(const Trajectory& traj){
        this->burst = traj.burst;
        this->B     = traj.B;
        this->S     = traj.S;
        this->C     = traj.C;
        this->Y     = traj.Y;

        return *this ;
}

double Trajectory::logp(const param* p, const constpar* constants){
    double logp = -0.5 * pow( B(0)/1e-4, 2) -0.5 * pow( (Y(0) - B(0)), 2)/p->sigma2 ;
    double dt = 1.0/constants->sampling_frequency;
    double W[2][2] = {{1-p->wbb[0]*dt, p->wbb[0]*dt},
        {p->wbb[1]*dt, 1-p->wbb[1]*dt}};
    double rate[2] = {p->r0*dt,p->r1*dt};

    for(unsigned int i=1;i<size;++i){
        
        logp += log(W[burst(i-1)][burst(i)]) +
                S(i)*log(rate[burst(i)])-rate[burst(i)] - log(gsl_sf_gamma(S(i)+1)) +
                -0.5*pow((B(i)-B(i-1))/(constants->bm_sigma*sqrt(dt)),2) +
                -0.5 * pow( (Y(i) - C(i) - B(i)), 2)/p->sigma2;

    }

    return logp;

}

        

// SMC class
// -------------------------------------------------------------------
//
SMC::SMC(string filename, int index, constpar& cst, bool has_header, int seed, unsigned int maxlen, string Gparam_file){
		
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (seed==0) ? cst.seed : seed);
		
    model = new GCaMP(cst.G_tot_mean,cst.gamma_mean,cst.DCaT_mean,cst.Rf_mean,cst.gam_in_mean, cst.gam_out_mean, Gparam_file);
    model->setTimeStepMode((TimeStepMode)cst.TSMode);

    arma::field <string> header;

    if(has_header){
        tracemat.load(arma::csv_name(filename, header));
    } else {
        tracemat.load(filename);
    }
    if(maxlen>0 && maxlen<tracemat.n_rows) tracemat = tracemat.rows(0,maxlen);

    data_time  = tracemat.col(0);
    data_y    = tracemat.col(index);
    constants = &cst;

    // The time units here are assumed to be seconds
    constants->sampling_frequency = 1.0 / (tracemat(1, 0)-tracemat(0,0));
    cout << "setting sampling frequency to: "<<constants->sampling_frequency << endl;
    constants->set_time_scales();

    // set number of particles
    nparticles=constants->nparticles;
    TIME = tracemat.n_rows;
    cout<<"nparticles: "<<nparticles<<endl;
    cout<<"TIME      : "<<TIME<<endl;

    // set particle system
    particleSystem.resize(TIME);
    for(unsigned int t=0;t<TIME;++t){
        particleSystem[t] = new Particle[nparticles];
        for(unsigned int j=0;j<nparticles;j++) particleSystem[t][j].index=j;
    }

}

SMC::SMC(arma::vec time, arma::vec data, int index, constpar& cst, bool has_header, int seed, unsigned int maxlen, string Gparam_file){
		
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (seed==0) ? cst.seed : seed);
		
    model = new GCaMP(cst.G_tot_mean,cst.gamma_mean,cst.DCaT_mean,cst.Rf_mean,cst.gam_in_mean, cst.gam_out_mean, Gparam_file);
    model->setTimeStepMode((TimeStepMode)cst.TSMode);

    arma::field <string> header;

    /*
    Stripping out un-needed calls to arma loading - this should more broadly prbably be re-worked such that the SMC constructor is more agnostic to the type of data (time, F) that is being passed in 

    if(has_header){
        tracemat.load(arma::csv_name(filename, header));
    } else {
        tracemat.load(filename);
    }
    if(maxlen>0 && maxlen<tracemat.n_rows) tracemat = tracemat.rows(0,maxlen);
    */
    //arma::vec data_time(time_vect);
    //std::copy(time_vect.begin(), time_vect.end(), data_time);
    //arma::vec data_y(data);
    //std::copy(data.begin(),data.end(),data_y);
    //data_time  = time_vect;
    //data_y    = data;
    constants = &cst;

    // The time units here are assumed to be seconds
    data_time = time;
    constants->sampling_frequency = 1.0 / (data_time(2)-data_time(1));
    cout << "setting sampling frequency to: "<<constants->sampling_frequency << endl;
    constants->set_time_scales();
    data_y = data;

    // set number of particles
    nparticles=constants->nparticles;

    TIME = data_time.n_elem;
    cout<<"nparticles: "<<nparticles<<endl;
    cout<<"TIME      : "<<TIME<<endl;

    // set particle system
    particleSystem.resize(TIME);
    for(unsigned int t=0;t<TIME;++t){
        particleSystem[t] = new Particle[nparticles];
        for(unsigned int j=0;j<nparticles;j++) particleSystem[t][j].index=j;
    }

}

SMC::SMC(arma::vec &Y, constpar& cst, bool v){

    verbose=v;

    constants = &cst;

    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng,constants->seed);

    data_time  = arma::regspace(0,Y.n_elem,1.0/constants->sampling_frequency);
    data_y    = Y;

    // The time units here are assumed to be seconds
    if(verbose) cout << "setting sampling frequency to: "<<constants->sampling_frequency << endl;
    constants->set_time_scales();

    // set number of particles
    nparticles=constants->nparticles;
    TIME = Y.n_rows;

    // set particle system
    particleSystem.resize(TIME);
    for(unsigned int t=0;t<TIME;++t){
        particleSystem[t] = new Particle[nparticles];
        for(unsigned int j=0;j<nparticles;j++) particleSystem[t][j].index=j;
    }
}


void SMC::rmu(Particle &p, double y, const param &par, bool set=false){

    if(!set){
        // draw state if not set

        p.burst = (gsl_rng_uniform(rng) > 0.5 ) ? 1 : 0;

        if(!constants->KNOWN_SPIKES){
            double rate=(p.burst==0) ? par.r0 : par.r1;
            p.S     = 0; //gsl_ran_poisson(rng, rate/constants->sampling_frequency);
        }

        p.B     = gsl_ran_gaussian(rng,1e-4);
    }

    p.C=model->initial_state;

    p.logWeight=-0.5*log(2*M_PI*par.sigma2)-0.5/par.sigma2*pow(y-0-p.B,2); // from g(y|x) only
                
}
double SMC::logf(const Particle &pin, const Particle &pout, const param &par){
    double lf = 0;
    double dt = 1.0/constants->sampling_frequency;
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    double rate[2] = {par.r0*dt,par.r1*dt};


    lf += log(W[pin.burst][pout.burst]) +
        pout.S*log(rate[pout.burst])-rate[pout.burst] - log(gsl_sf_gamma(pout.S+1)) +
        -0.5*pow((pout.B-pin.B)/(constants->bm_sigma*sqrt(dt)),2);

    return(lf);
}

// Here part.S is supposed to be given.
//
void SMC::move_and_weight_GTS(Particle &part, const Particle& parent, double y, const param &par, double g_noise, double u_noise, bool set=false){

    double probs[2];
    double log_probs[2];
    double Z;
    double ct;
    unsigned int burst;
    unsigned int ns=part.S;

    double dt = 1.0/constants->sampling_frequency;

    double rate[2] = {par.r0*dt,par.r1*dt};
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};

    double z[2] = {par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)),
                   dt*pow(constants->bm_sigma,2)/(par.sigma2+dt*pow(constants->bm_sigma,2))};
    double sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));
    
    arma::vec state_out(12);
    model->evolve_threadsafe(dt, (int)ns, parent.C, state_out, ct);
    
    log_probs[0] = log(W[parent.burst][0]);
    log_probs[0] += ns*log(rate[0]) - log(gsl_sf_gamma(ns+1)) -rate[0];
    log_probs[0] += -0.5/(par.sigma2+pow(constants->bm_sigma,2))*pow(y-ct-parent.B,2);
    log_probs[1] = log(W[parent.burst][1]);
    log_probs[1] += ns*log(rate[1]) - log(gsl_sf_gamma(ns+1)) -rate[1];
    log_probs[1] += -0.5/(par.sigma2+pow(constants->bm_sigma,2))*pow(y-ct-parent.B,2);

    utils::w_from_logW(log_probs,probs,2);
    Z=utils::Z_from_logW(log_probs,2);

    part.logWeight = log(Z);

    if(!set){
        // move particle if not already set
        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(2, probs);
        int idx = utils::gsl_ran_discrete_from_uniform(rdisc, u_noise);

        part.burst = idx;
        part.B     = z[0]*parent.B+z[1]*(y-parent.C(0)) + g_noise;

        gsl_ran_discrete_free(rdisc);
    }
    
    // part.C     = model->state;
    part.C = state_out;

}

void SMC::move_and_weight(Particle &part, const Particle& parent, double y, const param &par, double g_noise, double u_noise, bool set=false){

    //const int maxspikes = 2;  // The number of spikes goes from 0 to maxspikes-1
    const auto maxspikes = constants->maxspikes;
    double probs[2*maxspikes];
    double log_probs[2*maxspikes];
    double Z;
    double ct;
    unsigned int burst, ns;
    double dt = 1.0/constants->sampling_frequency;

    double rate[2] = {par.r0*dt,par.r1*dt};
    double W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    
    double z[2] = {par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)),
                   dt*pow(constants->bm_sigma,2)/(par.sigma2+dt*pow(constants->bm_sigma,2))};
    //double sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));
    
    arma::vec state_out(12); // <xxx flag for removal xxx>
    for(unsigned int i=0;i<2*maxspikes;i++){

        burst = floor(i/maxspikes);
        ns    = i%maxspikes;

        model->evolve_threadsafe(dt, (int)ns, parent.C, state_out, ct);
        
        log_probs[i] = log(W[parent.burst][burst]);
        log_probs[i] += ns*log(rate[burst]) - log(tgamma(ns+1)) -rate[burst];
        log_probs[i] += -0.5/(par.sigma2+pow(constants->bm_sigma,2))*pow(y-ct-parent.B,2); 
        
    } 

    utils::w_from_logW(log_probs,probs,2*maxspikes);
    Z=utils::Z_from_logW(log_probs,2*maxspikes);

    part.logWeight = log(Z);

    if(!set){
        // move particle if not already set
        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(2*maxspikes, probs);
        int idx = utils::gsl_ran_discrete_from_uniform(rdisc, u_noise);

        part.burst = floor(idx/maxspikes);
        part.S     = idx%maxspikes;

        //part.B     = z[0]*parent.B+z[1]*(y-model->getDFF(parent.C)) + gsl_ran_gaussian(rng,sigma_B_posterior);
        part.B     = parent.B + g_noise;

        gsl_ran_discrete_free(rdisc);
    }
    
    model->evolve_threadsafe(dt, part.S, parent.C, state_out, ct);
    part.C     = state_out;

}

void SMC::sampleParameters(const param &pin, param &pout, Trajectory &traj){

    // Sampling the posterior burst transition rates
    //
    int counts[2][2] = {{0,0},{0,0}};

    for(unsigned int t=1;t<traj.size;t++){
        counts[traj.burst(t-1)][traj.burst(t)]++;
    }
    double dt = 1.0/constants->sampling_frequency;

    double alpha_w01_post = constants->alpha_w01 + counts[0][1];
    double beta_w01_post  = constants->beta_w01  + counts[0][0]*dt;
    double alpha_w10_post = constants->alpha_w10 + counts[1][0];
    double beta_w10_post  = constants->beta_w10  + counts[1][1]*dt;

    pout.wbb[0] = gsl_ran_gamma(rng,alpha_w01_post,1.0/beta_w01_post);
    pout.wbb[1] = gsl_ran_gamma(rng,alpha_w10_post,1.0/beta_w10_post);

    // Sampling the posterior firing rates
    //
    arma::uvec which_b0 = arma::find(traj.burst==0);
    double alpha_rate_b0_post = constants->alpha_rate_b0 + arma::accu(traj.S(which_b0));
    double beta_rate_b0_post  = constants->beta_rate_b0  + which_b0.n_elem*dt;

    arma::uvec which_b1 = arma::find(traj.burst==1);
    double alpha_rate_b1_post = constants->alpha_rate_b1 + arma::accu(traj.S(which_b1));
    double beta_rate_b1_post  = constants->beta_rate_b1  + which_b1.n_elem*dt;

    pout.r0 = gsl_ran_gamma(rng,alpha_rate_b0_post,1.0/beta_rate_b0_post);
    pout.r1 = gsl_ran_gamma(rng,alpha_rate_b1_post,1.0/beta_rate_b1_post);

    // Sampling standard deviation
    //
    double res = pow(arma::norm(data_y - traj.B - traj.C),2);
    pout.sigma2 = 1.0 / gsl_ran_gamma(rng, constants->alpha_sigma2 + traj.size / 2.0, 1.0 / (constants->beta_sigma2 + 0.5 * res));
    //pout.sigma2=pin.sigma2;

    // Sampling Amax, rise and decay times

    unsigned int sampled_variable;
    sampled_variable=gsl_rng_uniform_int(rng,6);

    unsigned int var_selec[6] = {0,0,0,0,0,0}; 
    var_selec[sampled_variable] = 1;

    if(constants->KNOWN_SPIKES){
        var_selec[0]=1;
        var_selec[1]=1;
        var_selec[2]=1;
        var_selec[3]=1;
        var_selec[4]=1;
        var_selec[5]=1;
    }

    // Sampling parameter set using gaussian moves
    //double G_tot_test = pin.G_tot + var_selec[0]*gsl_ran_gaussian(rng,constants->G_tot_prop_sd);
    //double gamma_test = pin.gamma + var_selec[1]*gsl_ran_gaussian(rng,constants->gamma_prop_sd);
    //double DCaT_test  = pin.DCaT  + var_selec[2]*gsl_ran_gaussian(rng,constants->DCaT_prop_sd);
    //double Rf_test    = pin.Rf    + var_selec[3]*gsl_ran_gaussian(rng,constants->Rf_prop_sd);
    //double gam_in_test  = pin.gam_in  + var_selec[4]*gsl_ran_gaussian(rng,constants->gam_in_prop_sd);
    //double gam_out_test = pin.gam_out + var_selec[5]*gsl_ran_gaussian(rng,constants->gam_out_prop_sd);

    // Sampling parameters using gamma multiplicative proposals
    
    double proposal_factors[6] = {1,1,1,1,1,1};
    for(unsigned int i=0; i<6; i++){
        proposal_factors[i] = gsl_ran_gamma(rng,1000,0.001);
    }

    double G_tot_test = pin.G_tot*((1-var_selec[0])+var_selec[0]*proposal_factors[0]);
    double gamma_test = pin.gamma*((1-var_selec[1])+var_selec[1]*proposal_factors[1]);
    double DCaT_test  = pin.DCaT*((1-var_selec[2])+var_selec[2]*proposal_factors[2]);
    double Rf_test    = pin.Rf *((1-var_selec[3])+var_selec[3]*proposal_factors[3]);
    double gam_in_test  = pin.gam_in *((1-var_selec[4])+var_selec[4]*proposal_factors[4]);
    double gam_out_test = pin.gam_out*((1-var_selec[5])+var_selec[5]*proposal_factors[5]);
    
    arma::vec partest={G_tot_test,gamma_test,DCaT_test,Rf_test,gam_in_test,gam_out_test};
    string parnames[] = {"G_tot", "gamma","DCaT","Rf","gam_in","gam_out"};
    arma::vec C_test(traj.size);
    double res_test;

    // resetting the GCaMP model
    model->setParams(G_tot_test,gamma_test,DCaT_test,Rf_test,gam_in_test, gam_out_test);
    model->init();

    res_test=0;
    for(unsigned int t=0;t<traj.size;t++) {
        model->evolve(dt,traj.S(t));
        res_test += pow(data_y(t)-traj.B(t)-model->DFF,2);
        C_test(t) = model->DFF;
    }

    double log_alpha_MH = -0.5*pow((G_tot_test-constants->G_tot_mean)/constants->G_tot_sd,2) + 0.5*pow((pin.G_tot-constants->G_tot_mean)/constants->G_tot_sd,2) 
                          -0.5*pow((gamma_test-constants->gamma_mean)/constants->gamma_sd,2) + 0.5*pow((pin.gamma-constants->gamma_mean)/constants->gamma_sd,2)
                          -0.5*pow((DCaT_test-constants->DCaT_mean)/constants->DCaT_sd,2)   + 0.5*pow((pin.DCaT-constants->DCaT_mean)/constants->DCaT_sd,2)
                          -0.5*pow((Rf_test-constants->Rf_mean)/constants->Rf_sd,2)+ 0.5*pow((pin.Rf-constants->Rf_mean)/constants->Rf_sd,2) 
                          -0.5*pow((gam_in_test-constants->gam_in_mean)/constants->gam_in_sd,2)+ 0.5*pow((pin.gam_in-constants->gam_in_mean)/constants->gam_in_sd,2) 
                          -0.5*pow((gam_out_test-constants->gam_out_mean)/constants->gam_out_sd,2)+ 0.5*pow((pin.gam_out-constants->gam_out_mean)/constants->gam_out_sd,2) 
                          -0.5*(res_test)/pout.sigma2 + 0.5*(res/pout.sigma2);

    // add proposal factors
    for (unsigned int i=0; i<6;i++){
        //log_alpha_MH += stats::gsl_gamma_logpdf(1.0/proposal_factors[i],100,0.01) - stats::gsl_gamma_logpdf(proposal_factors[i],100,0.01);
    }

    cout<<"var: "<<setw(6)<<parnames[sampled_variable]<<' '
        <<"value: "<<setw(15)<<partest(sampled_variable)<<' '
        <<"logAlpha: "<<setw(10)<<log_alpha_MH<<endl;

    if(gsl_rng_uniform(rng) < exp(log_alpha_MH)) {
        pout.G_tot  = G_tot_test;
        pout.gamma  = gamma_test;
        pout.DCaT   = DCaT_test;
        pout.Rf     = Rf_test;
        pout.gam_in = gam_in_test;
        pout.gam_out= gam_out_test;
        traj.C      = C_test;
    } else {
        pout.G_tot  = pin.G_tot;
        pout.gamma  = pin.gamma;
        pout.DCaT   = pin.DCaT;
        pout.Rf     = pin.Rf;
        pout.gam_in = pin.gam_in;
        pout.gam_out= pin.gam_out;
    }


}

void SMC::PF(const param &par){

    // define particle system
    cout<<"TIME = "<<TIME<<endl;
    cout<<"nparticles = "<<nparticles<<endl;

    // define weights
    double w[nparticles],logW[nparticles];

    // initialize all particles at time 0
    for(unsigned int i=0;i<nparticles;++i){
        rmu(particleSystem[0][i],data_y(0),par);
    }

    vector<double> g_noise(nparticles);
    vector<double> u_noise(nparticles);
    vector<int> a_noise(nparticles);

    for(unsigned int t=1;t<TIME;++t){
        // Multinomial resampling
        // // // collect weights
        for(unsigned int i=0;i<nparticles;i++){
            logW[i] = particleSystem[t-1][i].logWeight;
        }

        utils::w_from_logW(logW,w,nparticles);

        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(nparticles, w);
        
        // Pregenerate noise so we can have noise that is not dependent on the thread execution order
        double dt = 1.0/constants->sampling_frequency;
        double sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));  
        for(unsigned int i=0;i<nparticles;i++){
            if(i != 0) {
                a_noise[i] = gsl_ran_discrete(rng,rdisc);
                u_noise[i] = gsl_rng_uniform(rng);
                g_noise[i] = gsl_ran_gaussian(rng,sigma_B_posterior);
            }
        }

        #pragma omp parallel for schedule(static)
        for(unsigned int i=0;i<nparticles;i++){
            
            //set ancestor of particle i
            
            particleSystem[t][i].ancestor = a_noise[i]; 
            particleSystem[t][i].logWeight = -log(nparticles);

            //move particle
            move_and_weight(particleSystem[t][i], particleSystem[t-1][a_noise[i]], data_y(t), par, g_noise[i], u_noise[i]);
        }

        gsl_ran_discrete_free(rdisc);

    }

    ofstream outfile_C("PF_output_C.dat");
    ofstream outfile_B("PF_output_B.dat");
    ofstream outfile_A("PF_output_ancestors.dat");
    ofstream outfile_test("PF_all_part.dat");
    ofstream outfile_S("PF_output_S.dat");


    for(unsigned int i=0;i<nparticles;i++){
        unsigned int a=i;
        unsigned int t=TIME;

        while(t>0){
            t--;
            outfile_A<<a<<' ';
            outfile_C<<particleSystem[t][a].C(0)<<' ';
            outfile_S<<particleSystem[t][a].S<<' ';
            outfile_B<<particleSystem[t][a].B<<' ';
            a=particleSystem[t][a].ancestor;
        }
        outfile_C<<endl;
        outfile_B<<endl;
        outfile_A<<endl;
        outfile_S<<endl;
    }

  
}

// A global to keep track whether Kokkos has been setup.
bool Kokkos_Initialized = false;

void init_kokkos()
{
	if (!Kokkos_Initialized) {
		Kokkos::initialize();
		Kokkos_Initialized = true;
	}
}
 
void SMC::PGAS(const param &par, const Trajectory &traj_in, Trajectory &traj_out){

    unsigned int t,a;
    int i;

    double dt=1.0/constants->sampling_frequency;

    // fixes omp complaints
    //omp_set_num_threads(1);
    
    // set model parameters
    model->setParams(par.G_tot,par.gamma,par.DCaT,par.Rf, par.gam_in, par.gam_out);

    // set particle 0 from input trajectory
    // Because trajectories do not store the full GCaMP state over time we have to re-evolve from the initial state.
    model->init();  // to initialize the GCaMP state to the steady state.
		
    for(t=0;t<TIME;t++){
        particleSystem[t][0].B=traj_in.B(t);
        particleSystem[t][0].burst=traj_in.burst(t);
        particleSystem[t][0].S=traj_in.S(t);

        model->evolve(dt,traj_in.S(t));
        particleSystem[t][0].C=model->state;
				
        particleSystem[t][0].ancestor=0;
				
    }

    particleSystem[0][0].ancestor = -1;

    
    // set spike count to all particles if training
    if(constants->KNOWN_SPIKES){
        for(t=0;t<TIME;t++){
            for(i=0;i<nparticles;i++){
                particleSystem[t][i].S=traj_in.S(t);
            }
        }
    }

    
    // define weights
    double w[nparticles],logW[nparticles];
    double ar_w[nparticles], ar_logW[nparticles];
	
    // initialize all particles at time 0 (excluded particle 0) and set all weights
    for(i=0;i<nparticles;++i){
        rmu(particleSystem[0][i],data_y(0),par,i==0); // i==0 allows to generate latent states only if i!=0
    }

    vector<double> g_noise(nparticles);
    vector<double> u_noise(nparticles);

    init_kokkos();
    
    // Copy the state of particles for all times to our structure of arrays on device memory
    ParticleArray particleArray(nparticles, TIME, constants->maxspikes);
    for(t=0;t<TIME;t++)
        for(i=0;i<nparticles;i++)
            particleArray.set_particle(t, i, particleSystem[t][i]);
    particleArray.copy_to_device();

    // Create a GPU view of the data_y vector
    VectorType data_y_view = VectorType("data_y_view", data_y.n_elem);
    VectorType::HostMirror data_y_h = Kokkos::create_mirror_view(data_y_view);
    for (unsigned int i = 0; i < data_y.n_elem; i++)
        data_y_h(i) = data_y(i);
    Kokkos::deep_copy(data_y_view, data_y_h);

    // Create the noise views
    VectorType g_noise_view = VectorType("g_noise_view", nparticles);
    VectorType u_noise_view = VectorType("u_noise_view", nparticles);
    VectorType::HostMirror g_noise_h = Kokkos::create_mirror_view(g_noise_view);
    VectorType::HostMirror u_noise_h = Kokkos::create_mirror_view(u_noise_view);

    GCaMP_params params;
    model->read_params(params);

    //bool die = false;
        
    for(t=1;t<TIME;++t){
        //if (die)
        //    break;

        //cout<<"    "<<t<<"      \r"<<flush;
        // Retrieve weights and calculate ancestor resampling weights for particle 0
        //#pragma omp parallel for schedule(static)
        
        //for(i=0;i<nparticles;i++){
        //    logW[i] = particleSystem[t-1][i].logWeight;
        //    ar_logW[i] = logW[i]+logf(particleSystem[t-1][i],particleSystem[t][0],par);
        //}

        particleArray.calc_ancestor_resampling(t, par, constants);

        // Check logW and ar_logW are equal on particleArray and particleSystem
        //for(i=0;i<nparticles;i++){
        //    double lw_cpu = logW[i];
        //    double lw_gpu = particleArray.logW_h(i);
        //    double ar_lw_cpu = ar_logW[i];
        //    double ar_lw_gpu = particleArray.ar_logW_h(i);
        //    if (fabs(lw_cpu - lw_gpu) > 1e-10 || fabs(ar_lw_cpu - ar_lw_gpu) > 1e-10) {
        //        cout << "Error: logW or ar_logW mismatch at particle " << i << " at time " << t << endl;
        //        cout << "CPU: logW = " << lw_cpu << ", ar_logW = " << ar_lw_cpu << endl;
        //        cout << "GPU: logW = " << lw_gpu << ", ar_logW = " << ar_lw_gpu << endl;
        //        die = true;
        //        break;
        //    }
        //}

        for(i=0;i<nparticles;i++){
            logW[i] = particleArray.logW_h(i);;
            ar_logW[i] = particleArray.ar_logW_h(i);
        }

        utils::w_from_logW(logW,w,nparticles);
        utils::w_from_logW(ar_logW,ar_w,nparticles);

        gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(nparticles, w);
        gsl_ran_discrete_t *ar_rdisc = gsl_ran_discrete_preproc(nparticles, ar_w);

        // Ancestor resampling of particle 0
        a = gsl_ran_discrete(rng,ar_rdisc);
        particleSystem[t][0].ancestor = a;
        
        // Resampling particles 1:nparticles
        for(i=1;i<nparticles;i++){
            a = gsl_ran_discrete(rng,rdisc);
            particleSystem[t][i].ancestor = a;
        }

        // Copy ancestors to device memory
        for(i=0;i<nparticles;i++)
            particleArray.new_ancestors_h(i) = particleSystem[t][i].ancestor;
        Kokkos::deep_copy(particleArray.new_ancestors, particleArray.new_ancestors_h);

        // // Pregenerate noise so we can have noise that is not dependent on the thread execution order
        // double dt = 1.0/constants->sampling_frequency;
        // double sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));  
        // for(i=0;i<nparticles;i++){
        //     if(i != 0) {
        //         u_noise[i] = gsl_rng_uniform(rng);
        //         g_noise[i] = gsl_ran_gaussian(rng,sigma_B_posterior);
        //     }
        // }
    
        // // Copy noise to device memory
        // for(i=0;i<nparticles;i++){
        //     g_noise_h(i) = g_noise[i];
        //     u_noise_h(i) = u_noise[i];
        // }
        // Kokkos::deep_copy(g_noise_view, g_noise_h);
        // Kokkos::deep_copy(u_noise_view, u_noise_h);

        // Move and weight particles
        //#pragma omp parallel for schedule(static)
        //for(i=0;i<nparticles;i++){
        //    a = particleSystem[t][i].ancestor;
        //    if(constants->KNOWN_SPIKES){
        //        move_and_weight_GTS(particleSystem[t][i], particleSystem[t-1][a], data_y(t), par, g_noise[i], u_noise[i], i==0);
        //    } else {
        //        move_and_weight(particleSystem[t][i], particleSystem[t-1][a], data_y(t), par, g_noise[i], u_noise[i], i==0);
        //    }
        //}

        // Copy back to host so we can theck things.
        particleArray.move_and_weight(t, data_y_view, par, constants, g_noise_view, u_noise, u_noise_view, params);

        //die = !particleArray.check_particle_system(t, particleSystem);

        gsl_ran_discrete_free(rdisc);
        gsl_ran_discrete_free(ar_rdisc);
        
    }

    // Copy the partilce system back to the CPU
    particleArray.copy_to_host();
    for(t=1;t<TIME;++t)
       for(i=0;i<nparticles;i++)
            particleArray.get_particle(t, i, particleSystem[t][i]);
		
    // Now use the last particle set to resample the new trajectory
    for(i=0;i<nparticles;i++){
        logW[i] = particleSystem[TIME-1][i].logWeight;
    }

    utils::w_from_logW(logW,w,nparticles);

    gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(nparticles, w);
    i = gsl_ran_discrete(rng,rdisc);

    t=TIME;

    //if(die)
    //    return;

    while(t>0){
        t--;
        traj_out.B(t)=particleSystem[t][i].B;
        traj_out.burst(t)=particleSystem[t][i].burst;
        traj_out.C(t)=model->getDFF(particleSystem[t][i].C);
        traj_out.S(t)=particleSystem[t][i].S;
        traj_out.Y(t)=data_y(t);
        //if(t>0) cout<<" -> "<<i<<" -> "<<particleSystem[t][i].ancestor;
        if(t>0){
            i=particleSystem[t][i].ancestor;
        }
    }
 
}
 
