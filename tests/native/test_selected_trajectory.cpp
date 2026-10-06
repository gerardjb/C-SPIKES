#include "selected_trajectory.h"
#include "GCaMP_model.h"
#include <iostream>
#if USE_GPU
using Exec=Kokkos::Cuda;
#else
using Exec=Kokkos::OpenMP;
#endif
using Space=Exec::memory_space;
using Matrix=Kokkos::View<double**,Space>;
using IntMatrix=Kokkos::View<int**,Space>;
using State=Kokkos::View<double**[12],Space>;
void require(bool b,const char* message) {if(!b) throw std::runtime_error(message);}
void fixture(int n,int time,GCaMP& model) {
    Matrix baseline("baseline",n,time);IntMatrix ancestor("ancestor",n,time),burst("burst",n,time),spikes("spikes",n,time);
    State calcium("calcium",n,time);
    auto bh=Kokkos::create_mirror_view(baseline);auto ah=Kokkos::create_mirror_view(ancestor);
    auto uh=Kokkos::create_mirror_view(burst),sh=Kokkos::create_mirror_view(spikes);
    auto ch=Kokkos::create_mirror_view(calcium);
    for(int t=0;t<time;++t) for(int i=0;i<n;++i) {
        bh(i,t)=0.25+i*7+t; uh(i,t)=(i+t)%2;sh(i,t)=(3*i+t)%4;
        ah(i,t)=t==0 ? -1 : (i+2*t)%n;
        for(int k=0;k<12;++k) ch(i,t,k)=(i*100+t*12+k)*1e-7;
    }
    Kokkos::deep_copy(baseline,bh);Kokkos::deep_copy(ancestor,ah);
    Kokkos::deep_copy(burst,uh);Kokkos::deep_copy(spikes,sh);Kokkos::deep_copy(calcium,ch);
    pgas::SelectedTrajectory<Exec> gather(time);
    for(int terminal=0;terminal<n;++terminal) {
        auto selected=gather.gather(terminal,ancestor,baseline,burst,calcium,spikes);
        int index=terminal;
        for(int t=time-1;t>=0;--t) {
            require(selected(t).baseline==bh(index,t),"baseline");
            require(selected(t).burst==uh(index,t),"burst");require(selected(t).spikes==sh(index,t),"spikes");
            arma::vec original(12),packet(12);
            for(int k=0;k<12;++k) {original(k)=ch(index,t,k);packet(k)=selected(t).calcium[k];}
            require(arma::all(original==packet),"all 12 components");
            require(model.getDFF(original)==model.getDFF(packet),"CPU fluorescence arithmetic");
            if(t>0) index=ah(index,t);
        }
    }
    bool rejected=false;
    try {gather.gather(n,ancestor,baseline,burst,calcium,spikes);}catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"out-of-range terminal");
    if(time>1) {
        ah(0,time-1)=n;Kokkos::deep_copy(ancestor,ah);rejected=false;
        try {gather.gather(0,ancestor,baseline,burst,calcium,spikes);}catch(const std::runtime_error&) {rejected=true;}
        require(rejected,"out-of-range ancestor");
    }
}
int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);int status=0;
    try {
        GCaMP model(6.04700454e-05,745.194622,1.44641501e-05,5.10813469,5.0730276513,5.01002741382,argv[1]);
        fixture(1,1,model);fixture(1,7,model);fixture(5,1,model);fixture(5,7,model);
        std::cout<<"PASS: fixed-history exact extraction, all terminals, particle 0, T=1, N=1, 12-state CPU getDFF, invalid indices\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';status=1;}
    Kokkos::finalize();return status;
}
