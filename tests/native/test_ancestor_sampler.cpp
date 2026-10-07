#include "ancestor_sampler.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#if USE_GPU
using Exec = Kokkos::Cuda;
#else
using Exec = Kokkos::OpenMP;
#endif
using Sampler = pgas::AncestorSampler<Exec>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void known_answers() {
    // Upstream https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    const uint32_t cases[3][10] = {
        {0,0,0,0,0,0,0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8},
        {0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,
         0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd},
        {0x243f6a88,0x85a308d3,0x13198a2e,0x03707344,0xa4093822,0x299f31d0,
         0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1}};
    Kokkos::View<int*, typename Exec::memory_space> failures("known_answers", 3);
    for (int i=0; i<3; ++i) {
        const auto c = cases[i];
        const pgas::PhiloxWords input{c[0],c[1],c[2],c[3]}, expected{c[6],c[7],c[8],c[9]};
        const uint32_t k0=c[4], k1=c[5];
        Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(i,i+1), KOKKOS_LAMBDA(int j) {
            const auto r=pgas::philox(input,k0,k1);
            failures(j) = r.x!=expected.x || r.y!=expected.y || r.z!=expected.z || r.w!=expected.w;
        });
    }
    auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), failures);
    for(int i=0;i<3;++i) require(h(i)==0,"Philox known answer");
}

void gsl_consumption() {
    auto a=gsl_rng_alloc(gsl_rng_mt19937), b=gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(a,123); gsl_rng_set(b,123);
    double w[]={0,1,3,0,9}; auto table=gsl_ran_discrete_preproc(5,w);
    for(int i=0;i<10003;++i) { gsl_ran_discrete(a,table); gsl_rng_uniform(b); }
    require(std::memcmp(gsl_rng_state(a),gsl_rng_state(b),gsl_rng_size(a))==0,"GSL consumption");
    gsl_ran_discrete_free(table); gsl_rng_free(a); gsl_rng_free(b);
}

void cdf_case(const std::vector<double>& logs, bool valid=true) {
    const int n=logs.size(); Sampler sampler(n);
    Sampler::Vector weights("weights",n); auto h=Kokkos::create_mirror_view(weights);
    for(int i=0;i<n;++i) h(i)=logs[i]; Kokkos::deep_copy(weights,h);
    sampler.build(weights,weights);
    bool caught=false;
    try { sampler.check(); } catch (const std::runtime_error&) { caught=true; }
    require(caught!=valid,"Invalid input policy");
    if(!valid) return;
    const auto cdf=sampler.cdf();
    // Test endpoints, exact boundaries, and interiors against serial CPU CDF.
    std::vector<double> oracle(n); double m=-INFINITY, total=0;
    for(auto x:logs) m=std::max(m,x);
    for(int i=0;i<n;++i) { total+=std::exp(logs[i]-m); oracle[i]=total; }
    std::vector<double> uniforms{0.0,std::nextafter(1.0,0.0),0.25,0.5,0.75};
    for(auto x:oracle) if(x<total) uniforms.push_back(x/total);
    Sampler::Vector u("u",uniforms.size()); auto uh=Kokkos::create_mirror_view(u);
    for(size_t j=0;j<uniforms.size();++j) uh(j)=uniforms[j]; Kokkos::deep_copy(u,uh);
    Sampler::Indices indices("indices",uniforms.size());
    Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0,uniforms.size()), KOKKOS_LAMBDA(int j) {
        indices(j)=pgas::invert_cdf(cdf,0,n,u(j));
    });
    auto ih=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),indices);
    for(size_t j=0;j<uniforms.size();++j) {
        double target=uniforms[j]*total;
        if(!(target<total)) target=std::nextafter(total,0.0);
        int expected=0; while(expected<n-1 && oracle[expected]<=target) ++expected;
        require(ih(j)==expected,"CPU CDF oracle");
        require(std::isfinite(logs[ih(j)]),"Zero mass selected");
    }
}

void distribution() {
    const int n=4, draws=200000;
    Sampler sampler(n); Sampler::Vector ordinary("ordinary",n), ref("ref",n);
    auto wh=Kokkos::create_mirror_view(ordinary), rh=Kokkos::create_mirror_view(ref);
    const double p[]={0.1,0.2,0.0,0.7}, q[]={0.7,0.0,0.2,0.1};
    for(int i=0;i<n;++i) {wh(i)=std::log(p[i]);rh(i)=std::log(q[i]);}
    Kokkos::deep_copy(ordinary,wh); Kokkos::deep_copy(ref,rh); sampler.build(ordinary,ref);
    auto cdf=sampler.cdf();
    Kokkos::View<int**,typename Exec::memory_space> counts("counts",4,4);
    Kokkos::View<int,typename Exec::memory_space> errors("errors");
    // Exercise production draw's routing once, then count 200k paired draws
    // without 200k host launches. Stream separation includes sweep high word.
    Sampler::Indices production("production",n); sampler.draw(production,101,0,1);
    auto ph=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),production);
    auto ch=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cdf);
    for(int j=0;j<n;++j) require(ph(j)==pgas::invert_cdf(ch,j==0?1:0,n,
        pgas::ancestor_uniform(101,0,1,j)),"Production routing");
    Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0,draws),KOKKOS_LAMBDA(int j) {
        const double u=pgas::ancestor_uniform(101,j,1,1);
        const double v=pgas::ancestor_uniform(101,j,1,0);
        if(!(u>=0 && u<1 && v>=0 && v<1) || u==v ||
           u==pgas::ancestor_uniform(102,j,1,1) ||
           u==pgas::ancestor_uniform(101,j,2,1) ||
           u==pgas::ancestor_uniform(101,uint64_t(j)+(UINT64_C(1)<<32),1,1))
            Kokkos::atomic_increment(&errors());
        const int a=pgas::invert_cdf(cdf,0,n,u), b=pgas::invert_cdf(cdf,1,n,v);
        Kokkos::atomic_increment(&counts(a,b));
    });
    sampler.check(); int bad=0; Kokkos::deep_copy(bad,errors); require(bad==0,"RNG ownership");
    auto c=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),counts);
    for(int a=0;a<n;++a) for(int b=0;b<n;++b) {
        const double prob=p[a]*q[b], expected=draws*prob;
        require(std::abs(c(a,b)-expected)<=6*std::sqrt(draws*prob*(1-prob))+2,"Joint frequencies");
    }
}

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int result=0;
    try {
        known_answers(); gsl_consumption();
        cdf_case({0}); cdf_case({0,0,0}); cdf_case({-INFINITY,0,-INFINITY});
        cdf_case({0,0,0,0,0,0,0}); cdf_case({-1e300,-1e300,-INFINITY});
        cdf_case({1000,0,-1000}); cdf_case({-INFINITY,-INFINITY},false);
        cdf_case({0,NAN},false); cdf_case({0,INFINITY},false); distribution();
        std::cout << "PASS: Philox KAT, GSL consumption, CDF oracle, invalid weights, routing, ownership, 200000 joint draws\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';result=1;}
    Kokkos::finalize();return result;
}
