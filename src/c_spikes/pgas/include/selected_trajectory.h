#ifndef C_SPIKES_SELECTED_TRAJECTORY_H
#define C_SPIKES_SELECTED_TRAJECTORY_H
#include <Kokkos_Core.hpp>
#include <stdexcept>
namespace pgas {
struct SelectedParticle {
    double baseline;
    double calcium[12];
    int burst, spikes;
};

// These are the only state fields required by CPU parameter sampling/output.
// Keep all twelve components so the existing CPU getDFF arithmetic is reused.
template<class ExecutionSpace>
class SelectedTrajectory {
public:
    using Space=typename ExecutionSpace::memory_space;
    using Packets=Kokkos::View<SelectedParticle*,Space>;
    using Indices=Kokkos::View<int*,Space>;
    using Error=Kokkos::View<int,Space>;
    explicit SelectedTrajectory(int time) : time_(time), path_("selected_path",time),
        packets_("selected_trajectory",time), error_("selected_path_error") {
        if(time<1) throw std::invalid_argument("Selected trajectory requires T >= 1");
    }

    template<class Ancestors,class Baseline,class Burst,class Calcium,class Spikes>
    typename Packets::HostMirror gather(int terminal,const Ancestors& ancestors,
        const Baseline& baseline,const Burst& burst,const Calcium& calcium,const Spikes& spikes) {
        const int n=ancestors.extent(0), time=time_;
        if(terminal<0 || terminal>=n || int(ancestors.extent(1))!=time)
            throw std::invalid_argument("Selected trajectory dimensions or terminal index invalid");
        const auto path=path_;const auto packets=packets_;const auto error=error_;
        Kokkos::parallel_for("selected_backtrace",Kokkos::RangePolicy<ExecutionSpace>(0,1),
            KOKKOS_LAMBDA(int) {
                int index=terminal;
                error()=0;
                for(int t=time-1;t>=0;--t) {
                    if(index<0 || index>=n) {error()=1;index=0;}
                    path(t)=index;
                    if(t>0) index=ancestors(index,t);
                }
            });
        // Both kernels use the same default execution instance. The backtrace
        // completes before gather; the synchronous deep copies complete both.
        Kokkos::parallel_for("selected_gather",Kokkos::RangePolicy<ExecutionSpace>(0,time),
            KOKKOS_LAMBDA(int t) {
                const int index=path(t);
                packets(t).baseline=baseline(index,t);
                packets(t).burst=burst(index,t);
                packets(t).spikes=spikes(index,t);
                for(int k=0;k<12;++k) packets(t).calcium[k]=calcium(index,t,k);
            });
        int invalid=0;Kokkos::deep_copy(invalid,error);
        if(invalid) throw std::runtime_error("Selected trajectory encountered invalid ancestor");
        return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),packets);
    }
private:
    int time_;
    Indices path_;
    Packets packets_;
    Error error_;
};
}
#endif
