#include "include/particle_array.h"
#include "include/constants.h"
#include "include/param.h"
#include "include/particle.h"
#include "include/GCaMP_model.h"
#include "include/utils.h"
#include <Kokkos_Random.hpp>

using namespace std;

template<unsigned int N>
KOKKOS_FUNCTION
void w_from_logW(const RowVectorType src, RowVectorType out){

    // Get the maximum log weight for the particle
    Scalar max_log_prob = -INFINITY;
    for(int i=0;i<N;i++) 
        if (src(i) > max_log_prob) 
            max_log_prob = src(i);
                    
    // Compute w (probs)
    for(int i=0;i<N;i++)
        out(i) = exp(src(i)-max_log_prob);
        
}

// A helper function that generates a discrete random variable from a uniform sample.
// This implmenents the naive linear scan method. See:
// https://publikationen.bibliothek.kit.edu/1000133378/115756109#:~:text=With%20weighted%20random%20sampling%2C%20each,1)%20is%20the%20alias%20table.
// for a good description of this method and others to do this more efficiently.
// I think for small N where we are only doing this once per particle, this is fine. 
//template<unsigned int N>
KOKKOS_FUNCTION
unsigned int discrete_from_uniform(int N, Scalar u, RowVectorType probs) {
    Scalar total = 0.0;
    for (int i = 0; i < N; ++i) total += probs(i);

    Scalar cumulative = 0.0;
    for (int i = 0; i < N; ++i) {
        cumulative += probs(i) / total;
        if (u < cumulative) return i;
    }
    return static_cast<unsigned int>(N - 1);  // catch any round-off
}

KOKKOS_FUNCTION

Scalar fixedStep_LA_kernel(
    Scalar deltat, int ns,
    const StateVectorType state_in,
    StateVectorType state_out,
    const GCaMP_params & p,
    bool save_state)
{

    Scalar G[9];

    for(int i=0;i<9;i++) G[i] = state_in(i);

    Scalar BCa = state_in(9);
    Scalar dBCa_dt;

    Scalar Ca  = state_in(10);
    Scalar Ca_in = state_in(11);
    Scalar dCa_dt, dCa_in_dt;
    
    Scalar Gflux, Cflux;
    Scalar finedt=100e-6;
    Scalar dt;

    Scalar calcium_input;

    Scalar konN,   koffN;
    Scalar konC,   koffC;
    Scalar konPN,  koffPN;
    Scalar konPN2, koffPN2;
    Scalar konPC,  koffPC;
    Scalar konPC2,  koffPC2;

    // Load from the params struct
    koffN = p.koffN;
    koffC = p.koffC;
    konPN = p.konPN;
    koffPN = p.koffPN;
    konPN2 = p.konPN2;
    koffPN2 = p.koffPN2;
    konPC = p.konPC;
    koffPC = p.koffPC;
    konPC2 = p.konPC2;
    koffPC2 = p.koffPC2;
    
    // Calculate konN and konC
    Scalar logCa = log(Ca);
    konN = p.Gparams[0]*exp(p.Gparams[2]*logCa);
    konC = p.Gparams[3]*exp(p.Gparams[5]*logCa);
  
    // Intiliaze the Gmatrix
	Scalar Gmatrix[9][9] = {{-(konN+konC),koffN,koffPN,0,koffC,koffPC,0,0,0},
        {konN,-(koffN+konPN+konC),0,0,0,0,koffPC2,koffC,0},
        {0,konPN,-(konC+koffPN),koffC,0,0,0,0,koffPC2},
        {0,0,konC,-(koffC+konPC2+koffPN2 ),0,0,0,konPN2,0},
        {konC,0,0,koffPN2,-(koffC+konPC+konN),0,0,koffN,0},
        {0,0,0,0,konPC,-(koffPC+konN),koffN,0,koffPN2},
        {0,0,0,0,0,konN,-(koffN+konPN2+koffPC2),konPC2,0},
        {0,konC,0,0,konN,0,0,-(koffC+koffN+konPC2+konPN2),0},
        {0,0,0,konPC2,0,0,konPN2,0,-(koffPN2+koffPC2)}};

    Scalar dG_dt[9];
    Scalar old_t = 0;
    Scalar t = finedt;
    int tstep_i = 1;
    int n_steps = floor(deltat/finedt);
    while(t <= (n_steps*finedt)) {

        calcium_input = (tstep_i==1) ? ns*p.DCaT/finedt : 0;

        Scalar logCa = log(Ca);
	    konN = p.Gparams[0]*exp(p.Gparams[2]*logCa);
        konC = p.Gparams[3]*exp(p.Gparams[5]*logCa);

        // Update the G matrix
        Gmatrix[0][0] = -(konN+konC);
        Gmatrix[1][0] = konN;
        Gmatrix[1][1] = -(koffN+konPN+konC);
        Gmatrix[2][2] = -(konC+koffPN);
        Gmatrix[3][2] = konC;
        Gmatrix[4][0] = konC;
        Gmatrix[4][4] = -(koffC+konPC+konN);
        Gmatrix[5][5] = -(koffPC+konN);
        Gmatrix[6][5] = konN;
        Gmatrix[7][1] = konC;
        Gmatrix[7][4] = konN;
       
        // arma::vec dG_dt  = Gmatrix*G;
        // Implement matrix vector product
        for(int i=0;i<9;i++){
            dG_dt[i] = 0;
            for(int j=0;j<9;j++) dG_dt[i] += Gmatrix[i][j]*G[j];
        }

        // Gflux = flux_konN_konC(konN, konC, G);
        // Calculate fluxes per binding site
        Scalar N = -(konN*(G[0] + G[4] + G[5])) +
                koffN*(G[1] + G[6]+ G[7])  + 
                koffPN*G[2]                + 
                koffPN2*(G[3] + G[8]);

        Scalar C = -(konC *(G[0] + G[1] + G[2])) +
                koffC*(G[4] + G[3] + G[7])  + 
                koffPC*(G[5])+ 
                koffPC2*(G[6]+G[8]);

        int Csites=2;
        Gflux = 2*N + Csites*C;

        Cflux   = -p.gamma*(Ca-p.c0) + Gflux;
        dBCa_dt = Cflux*p.kapB/(p.kapB + 1);
        dCa_in_dt = p.gam_in*(Ca-p.c0) - p.gam_out*(Ca_in - p.c0);

        dCa_dt  = -p.gamma*(Ca-p.c0) // pump out
            -p.gam_in*(Ca - p.c0) + p.gam_out*(Ca_in - p.c0) //intra-compartmental exchange
            + Gflux - dBCa_dt + calcium_input*1/(p.kapB + 1);

        dt  = t-old_t;
        
        // G   = G   +  dt*dG_dt;
        for(int i=0;i<9;i++) G[i] = G[i] + dt*dG_dt[i];

        BCa = BCa +  dt*(dBCa_dt+p.kapB/(p.kapB+1)*calcium_input);
        Ca  = Ca  +  dt*dCa_dt;
        Ca_in = Ca_in + dt*dCa_in_dt;
        old_t = t; 
        t = finedt*tstep_i;
        tstep_i++;
    }

    if (save_state) {
        for(unsigned int i=0;i<9;i++) state_out(i) = G[i];
        state_out(9)  = BCa;
        state_out(10) = Ca;
        state_out(11) = Ca_in;
    }

    int brightStates[5] = {2, 3, 5, 6, 8};
    Scalar brightStatesSum = 0.0;
    for(unsigned int i=0;i<5;i++) brightStatesSum += G[brightStates[i]];

    Scalar DFF_out = (brightStatesSum - p.Ginit)/(p.Ginit-p.G0+(p.Gsat-p.G0)/(p.Rf-1));

    return DFF_out;
}


ParticleArray::ParticleArray(int N, int T, int maxspikes) : N(N), T(T), maxspikes(maxspikes) 
{

    random_pool = RandPoolType(42);

    B = MatrixType("B",N,T);
    burst = IntMatrixType("burst",N,T);
    C = StateMatrixType("C",N,T);
    S = IntMatrixType("S",N,T);
    logWeight = MatrixType("logWeight",N,T);
    ancestor = IntMatrixType("ancestor",N,T);
    index = IntMatrixType("index",N,T);

    B_h = Kokkos::create_mirror_view(B);
    burst_h = Kokkos::create_mirror_view(burst);
    C_h = Kokkos::create_mirror_view(C);
    S_h = Kokkos::create_mirror_view(S);
    logWeight_h = Kokkos::create_mirror_view(logWeight);
    ancestor_h = Kokkos::create_mirror_view(ancestor);
    index_h = Kokkos::create_mirror_view(index);

    // Intermediate arrays for the move and weight steps
    log_probs = MatrixType("log_probs",N,maxspikes*2);
    log_probs_h = Kokkos::create_mirror_view(log_probs);

    probs = MatrixType("probs",N,maxspikes*2);
    probs_h = Kokkos::create_mirror_view(probs);

    discrete = IntVectorType("discrete",N);
    discrete_h = Kokkos::create_mirror_view(discrete);

    logW = VectorType("logW",N);
    logW_h = Kokkos::create_mirror_view(logW);

    ar_logW = VectorType("ar_logW",N);
    ar_logW_h = Kokkos::create_mirror_view(ar_logW);

    new_ancestors = IntVectorType("new_ancestors",N);
    new_ancestors_h = Kokkos::create_mirror_view(new_ancestors);
}


void ParticleArray::set_particle(int t, int idx, const Particle &p)
{
    B_h(idx,t) = p.B;
    burst_h(idx,t) = p.burst;
    for(int i=0;i<12;i++) C_h(idx,t,i) = p.C(i);
    S_h(idx,t) = p.S;
    logWeight_h(idx,t) = p.logWeight;
    ancestor_h(idx,t) = p.ancestor;
    index_h(idx,t) = p.index;
}

void ParticleArray::get_particle(int t, int idx, Particle &p)
{
    p.B = B_h(idx,t);
    p.burst = burst_h(idx,t);
    for(int i=0;i<12;i++) p.C(i) = C_h(idx,t,i);
    p.S = S_h(idx,t);
    p.logWeight = logWeight_h(idx,t);
    p.ancestor = ancestor_h(idx,t);
    p.index = index_h(idx,t);
}

void ParticleArray::copy_to_device()
{
    Kokkos::deep_copy(B,B_h);
    Kokkos::deep_copy(burst,burst_h);
    Kokkos::deep_copy(C,C_h);
    Kokkos::deep_copy(S,S_h);
    Kokkos::deep_copy(logWeight,logWeight_h);
    Kokkos::deep_copy(ancestor,ancestor_h);
    Kokkos::deep_copy(index,index_h);
}

void ParticleArray::copy_to_host()
{
    Kokkos::deep_copy(B_h,B);
    Kokkos::deep_copy(burst_h,burst);
    Kokkos::deep_copy(C_h,C);
    Kokkos::deep_copy(S_h,S);
    Kokkos::deep_copy(logWeight_h,logWeight);
    Kokkos::deep_copy(ancestor_h,ancestor);
    Kokkos::deep_copy(index_h,index);
}

bool ParticleArray::check_particle_system(int t, std::vector<Particle*> & particleSystem)
{
    copy_to_host();

    bool die = false;

    // Check that the particleArray and particleSystem are the same
    for(int i=0;i<N;i++){
        Particle p_cpu = particleSystem[t][i];
        Particle p_gpu;
        get_particle(t, i, p_gpu);
        if (p_cpu.S != p_gpu.S) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU S=" << p_cpu.S <<  " GPU S=" << p_gpu.S << endl;
            die = true;
        }
        if (fabs(p_cpu.B - p_gpu.B) > 1e-10){
            cout << "TIME=" << t << "  Particle(" << i << "): CPU B=" << p_cpu.B <<  " GPU B=" << p_gpu.B << endl;
            die = true;
        }
        if (p_cpu.burst != p_gpu.burst) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU burst=" << p_cpu.burst <<  " GPU burst=" << p_gpu.burst << endl;
            die = true;
        }
        for(int j=0;j<12;j++){
            if (fabs(p_cpu.C(j) - p_gpu.C(j)) > 1e-10) {
                cout << "TIME=" << t << "  Particle(" << i << "): CPU C(" << j << ")=" << p_cpu.C(j) <<  " GPU C(" << j << ")=" << p_gpu.C(j) << endl;
                die = true;
            }
        }
        if (p_cpu.ancestor != p_gpu.ancestor) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU ancestor=" << p_cpu.ancestor <<  " GPU ancestor=" << p_gpu.ancestor << endl;
            die = true;
        }
        if (fabs(p_cpu.logWeight - p_gpu.logWeight) > 1e-10) {
            cout << "TIME=" << t << "  Particle(" << i << "): CPU logWeight=" << p_cpu.logWeight <<  " GPU logWeight=" << p_gpu.logWeight << endl;
            die = true;
        }

        if (die)
            break;
    }

    return !die;
}

KOKKOS_FUNCTION
Scalar ParticleArray::logf(
    int part_idx_in, int t_in, 
    int part_idx_out, int t_out, 
    const param &par, constpar *constants) 
{
    Scalar lf = 0;
    Scalar dt = 1.0/constants->sampling_frequency;
    Scalar W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    Scalar rate[2] = {par.r0*dt,par.r1*dt};

    int pin_burst = burst(part_idx_in, t_in);
    Scalar pin_B = B(part_idx_in, t_in);
    int pin_S = S(part_idx_in, t_in);
    
    int pout_burst = burst(part_idx_out, t_out);
    Scalar pout_B = B(part_idx_out, t_out);
    int pout_S = S(part_idx_out, t_out);

    lf += log(W[pin_burst][pout_burst]) +
        pout_S*log(rate[pout_burst])-rate[pout_burst] - log(tgamma(pout_S+1)) +
        -0.5*pow((pout_B-pin_B)/(constants->bm_sigma*sqrt(dt)),2);

    return(lf);
}

void ParticleArray::calc_ancestor_resampling(
    int t, const param &par, constpar *constants)
{
    Scalar sampling_frequency = constants->sampling_frequency;
    Scalar wbb[2];
    wbb[0] = par.wbb[0];
    wbb[1] = par.wbb[1];
    Scalar r0 = par.r0;
    Scalar r1 = par.r1;
    Scalar bm_sigma = constants->bm_sigma;

    // Retrieve weights and calculate ancestor resampling weights
    Kokkos::parallel_for("calc_ancestor_weights",
        Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_CLASS_LAMBDA(const int part_idx) {
                
                Scalar lf = 0;
                Scalar dt = 1.0/sampling_frequency;
                Scalar W[2][2] = {{1-wbb[0]*dt, wbb[0]*dt},
                    {wbb[1]*dt, 1-wbb[1]*dt}};
                Scalar rate[2] = {r0*dt,r1*dt};

                int part_idx_in = part_idx;
                int t_in = t-1;
                int part_idx_out = 0;
                int t_out = t;

                int pin_burst = burst(part_idx_in, t_in);
                Scalar pin_B = B(part_idx_in, t_in);
                int pin_S = S(part_idx_in, t_in);
                
                int pout_burst = burst(part_idx_out, t_out);
                Scalar pout_B = B(part_idx_out, t_out);
                int pout_S = S(part_idx_out, t_out);

                if (part_idx == 0) {
                    // Kokkos::printf("logWeight=%f\n", logWeight(part_idx, t-1));
                    // Kokkos::printf("log(W[pin_burst][pout_burst])=%f\n", log(W[pin_burst][pout_burst]));
                    // Kokkos::printf("pin_burst=%d, pin_B=%f pin_S=%d\n", pin_burst, pin_B, pin_S);
                    // // Kokkos::printf("pout_burst=%d, pout_B=%f pout_S=%d\n", pout_burst, pout_B, pout_S);
                }

                lf += log(W[pin_burst][pout_burst]) +
                    pout_S*log(rate[pout_burst])-rate[pout_burst] - log(tgamma(pout_S+1)) +
                    -0.5*pow((pout_B-pin_B)/(bm_sigma*sqrt(dt)),2);

                logW(part_idx) = logWeight(part_idx, t-1);
                ar_logW(part_idx) = logWeight(part_idx, t-1) + lf;

                // auto generator = random_pool.get_state();
                // int rand_a = Kokkos::rand<RandGenType, int>::draw(generator, 0, N-1);
                // ancestor(part_idx, t) = rand_a ;
                // random_pool.free_state(generator);
                
            });
    Kokkos::fence("fence_calc_ancestor_weights");

    // Copy back to host (can remove this later when random number generation is done on GPU)
    Kokkos::deep_copy(logW_h, logW);
    Kokkos::deep_copy(ar_logW_h, ar_logW);
}

void ParticleArray::move_and_weight(
    int t,
    VectorType y, 
    const param &par, constpar *constants, 
    VectorType g_noise, std::vector<double> & u_noise,
    VectorType u_noise_view,
    const GCaMP_params & params)
{
    const int maxspikes = this->maxspikes;
    Scalar dt = 1.0/constants->sampling_frequency;

    Scalar rate[2] = {par.r0*dt,par.r1*dt};
    Scalar W[2][2] = {{1-par.wbb[0]*dt, par.wbb[0]*dt},
        {par.wbb[1]*dt, 1-par.wbb[1]*dt}};
    Scalar sigma_B_posterior = sqrt(dt*pow(constants->bm_sigma,2)*par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)));  
    Scalar z[2] = {par.sigma2/(par.sigma2+dt*pow(constants->bm_sigma,2)),
                   dt*pow(constants->bm_sigma,2)/(par.sigma2+dt*pow(constants->bm_sigma,2))};

    Scalar bm_sigma = constants->bm_sigma;

    // We don't actually need this value, but we need to pass it to the kernel
    // save_state=false will make sure it isn't written to, which would cause 
    // problems because it is the same address for all particles.
    //StateMatrixType state_out_tmp("state_out_tmp", 1, 1, 12);
    // StateVectorType state_out("state_out", 12);
    StateVectorType state_out = Kokkos::subview(C, 0, 0, Kokkos::ALL);

    Kokkos::parallel_for("evolve_for_maxspikes",
		Kokkos::RangePolicy<ExecSpace>(0, N*maxspikes*2),
            KOKKOS_CLASS_LAMBDA(const int idx) {
                int particle_idx = idx / (maxspikes*2);
                int spike_idx = idx % (maxspikes*2);
                int b = floor(spike_idx/maxspikes);
                int ns    = spike_idx % maxspikes;

                // Update the ancestors that we generated on the CPU
                ancestor(particle_idx, t) = new_ancestors(particle_idx);
             
                int a = ancestor(particle_idx, t);
                int parent_b = burst(a, t-1);
                Scalar parent_B = B(a, t-1);
                
                // Evolve
                // model->evolve_threadsafe(dt, (int)ns, parent.C, state_out, ct);
                StateVectorType state_in = Kokkos::subview(C, a, t-1, Kokkos::ALL);
                Scalar ct = fixedStep_LA_kernel(dt, ns, state_in, state_out, params, false);

                Scalar log_prob_tmp = log(W[parent_b][b]);
                log_prob_tmp += ns*log(rate[b]) - log(tgamma(ns+1)) - rate[b];
                log_prob_tmp += -0.5/(par.sigma2+pow(bm_sigma,2))*pow(y(t)-ct-parent_B,2);
                log_probs(particle_idx, spike_idx) = log_prob_tmp;

            });

    // // Wait for kernel to complete
    Kokkos::fence("fence_evolve_for_maxspikes");

    // Calclulate w and Z
    Kokkos::parallel_for("w_and_Z",
        Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_CLASS_LAMBDA(const int part_idx) {
                                
                // Get the maximum log weight for the particle
                Scalar max_log_prob = -INFINITY;
                for(int i=0;i<2*maxspikes;i++) 
                    if (log_probs(part_idx,i) > max_log_prob) 
                        max_log_prob = log_probs(part_idx, i);
                    
                // Compute w (probs)
                Scalar Z_tmp = 0;
                for(int i=0;i<2*maxspikes;i++) {
                    probs(part_idx, i) = exp(log_probs(part_idx, i)-max_log_prob);
                    Z_tmp += probs(part_idx, i);
                }

                //  normalization constant Z for log-probabilities
                Z_tmp = Z_tmp*exp(max_log_prob);
                logWeight(part_idx, t) = log(Z_tmp);

            });

    // // Wait for kernel to complete
    Kokkos::fence("fence_w_and_Z");

    // // Copy probabilities to host so we can generate the discrete random variables
    // // Will do this on GPU in the future after testing
    // IntVectorType discrete = IntVectorType("discrete", N);
    // IntVectorType::HostMirror discrete_h = Kokkos::create_mirror_view(discrete);

    // MatrixType::HostMirror probs_h = Kokkos::create_mirror_view(probs);
    // Kokkos::deep_copy(probs_h, probs);
    // for(int i=0;i<N;i++) {
    //     Scalar probs_tmp[2*maxspikes];
    //     for(int j=0;j<2*maxspikes;j++) probs_tmp[j] = probs_h(i,j);
    //     gsl_ran_discrete_t *rdisc = gsl_ran_discrete_preproc(2*maxspikes, probs_tmp);
    //     discrete_h(i) = utils::gsl_ran_discrete_from_uniform(rdisc, u_noise[i]);
    //     gsl_ran_discrete_free(rdisc);
    // }
    // Kokkos::deep_copy(discrete, discrete_h);

    // Move the particles
    Kokkos::parallel_for("move_particles",
        Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_CLASS_LAMBDA(const int part_idx) {

                // move particle if not already set
                if(part_idx != 0){
                    auto generator = random_pool.get_state();

                    Scalar u = Kokkos::rand<RandGenType, Scalar>::draw(generator, 0.0, 1.0);
                    int idx = discrete_from_uniform(2*maxspikes, u, Kokkos::subview(probs, part_idx, Kokkos::ALL));
                    // int idx = discrete(part_idx);

                    burst(part_idx, t) = floor(idx/maxspikes);
                    S(part_idx, t) = idx%maxspikes;

                    // B(part_idx, t) = B(ancestor(part_idx, t), t-1) + g_noise(part_idx);
                    B(part_idx, t) = B(ancestor(part_idx, t), t-1) + generator.normal(0.0, sigma_B_posterior);

                    // do not forget to release the state of the engine
                    random_pool.free_state(generator);
                }
                
                // model->evolve_threadsafe(dt, part.S, parent.C, state_out, ct);
                // part.C     = state_out;
                StateVectorType state_in = Kokkos::subview(C, ancestor(part_idx, t), t-1, Kokkos::ALL);
                StateVectorType state_out = Kokkos::subview(C, part_idx, t, Kokkos::ALL);
                Scalar ct = fixedStep_LA_kernel(dt, S(part_idx, t), state_in, state_out, params, true);

            });
    Kokkos::fence("fence_move_particles");

}