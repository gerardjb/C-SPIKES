#ifndef PARTICLE_ARRAY_H
#define PARTICLE_ARRAY_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "param.h"
#include "constants.h"
#include "GCaMP_model.h"

// Define the scalar type for the Kokkos views
using Scalar = double;

#define USE_GPU

#ifndef USE_GPU
// using ExecSpace = Kokkos::Serial;
// using MemSpace = Kokkos::HostSpace;
    using ExecSpace = Kokkos::OpenMP;
    using MemSpace = Kokkos::HostSpace;
#else
    using ExecSpace = Kokkos::Cuda;
    using MemSpace = Kokkos::CudaSpace;
#endif

using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

// Define the view types
using VectorType = Kokkos::View<Scalar*, MemSpace>;
using MatrixType = Kokkos::View<Scalar**, MemSpace>;
using IntVectorType = Kokkos::View<int*, MemSpace>;
using IntMatrixType = Kokkos::View<int**, MemSpace>;

// Define a subview row type
using RowVectorType = Kokkos::Subview<MatrixType, int, decltype(Kokkos::ALL)>;

// Define the random number generator type
using RandPoolType = Kokkos::Random_XorShift64_Pool<DeviceType>;
using RandGenType = RandPoolType::generator_type;

// Our state has a fixed size of 12, lets make a type for it. This willAdd commentMore actions
// be a 3D array with the first dimension being the number of particles,
// the second is time, and the third is the state variables
using StateMatrixType = Kokkos::View<Scalar**[12], MemSpace>;

// Lets make a subview for a single particle
using StateVectorType = Kokkos::Subview<StateMatrixType, int, int, decltype(Kokkos::ALL)>;

class Particle;

constexpr int maxspikes = 2;

// A class to store N particles over time. First dimension of all arrays is time, second is particle index
class ParticleArray
{
    public:

        MatrixType B;
        IntMatrixType burst;
        StateMatrixType C;
        IntMatrixType S;
        MatrixType logWeight;
        IntMatrixType ancestor;
        IntMatrixType index;

        // Host mirrors
        MatrixType::HostMirror B_h;
        IntMatrixType::HostMirror burst_h;
        StateMatrixType::HostMirror C_h;
        IntMatrixType::HostMirror S_h;
        MatrixType::HostMirror logWeight_h;
        IntMatrixType::HostMirror ancestor_h;
        IntMatrixType::HostMirror index_h;

        ParticleArray(int N, int T);

        void set_particle(int t, int idx, const Particle &p);
        void get_particle(int t, int idx, Particle &p);

        void copy_to_device();
        void copy_to_host();

        void move_and_weight(int t, VectorType y, const param &par, constpar *constants, 
                             VectorType g_noise, std::vector<double> & u_noise, VectorType u_noise_view, const GCaMP_params & params); 

        void calc_ancestor_resampling(int t, const param &par, constpar *constants);

        KOKKOS_FUNCTION
        Scalar logf(int part_idx_in, int t_in, int part_idx_out, int t_out, const param &par, constpar *constants);

        bool check_particle_system(int t, std::vector<Particle*> & particleSystem);

        // Number of particles
        int N;

        // Number of time points
        int T;
        
        // Random number generator requirement
        RandPoolType random_pool;

    // private:

        MatrixType log_probs;
        MatrixType::HostMirror log_probs_h;

        MatrixType probs;
        MatrixType::HostMirror probs_h;

        IntVectorType discrete;
        IntVectorType::HostMirror discrete_h;

        VectorType logW;
        VectorType::HostMirror logW_h;

        VectorType ar_logW;
        VectorType::HostMirror ar_logW_h;

        // Random noise for each particle
        IntVectorType new_ancestors;
        IntVectorType::HostMirror new_ancestors_h;

};

#endif