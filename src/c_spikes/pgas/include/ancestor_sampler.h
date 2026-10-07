#ifndef C_SPIKES_ANCESTOR_SAMPLER_H
#define C_SPIKES_ANCESTOR_SAMPLER_H

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <stdexcept>

namespace pgas {

// Philox4x32-10 (Salmon et al., SC11). Known-answer tests use Random123's
// published vectors. Unsigned arithmetic, including wraparound, is deliberate.
struct PhiloxWords { uint32_t x, y, z, w; };

KOKKOS_INLINE_FUNCTION
PhiloxWords philox(PhiloxWords c, uint32_t k0, uint32_t k1) {
    for (int round = 0; round < 10; ++round) {
        const uint64_t p0 = uint64_t(0xd2511f53U) * c.x;
        const uint64_t p1 = uint64_t(0xcd9e8d57U) * c.z;
        c = {uint32_t(p1 >> 32) ^ c.y ^ k0, uint32_t(p1),
             uint32_t(p0 >> 32) ^ c.w ^ k1, uint32_t(p0)};
        k0 += 0x9e3779b9U;
        k1 += 0xbb67ae85U;
    }
    return c;
}

KOKKOS_INLINE_FUNCTION
double ancestor_uniform(uint64_t seed, uint64_t sweep, uint32_t time, uint32_t particle) {
    // Domain reserved for ancestor selection; no mutable/shared RNG pool.
    seed ^= UINT64_C(0x50474153414e4353);
    const auto r = philox({particle, time, uint32_t(sweep), uint32_t(sweep >> 32)},
                          uint32_t(seed), uint32_t(seed >> 32));
    return double((uint64_t(r.x) << 21) | (r.y >> 11)) * 0x1.0p-53;
}

template<class CDF>
KOKKOS_INLINE_FUNCTION
int invert_cdf(const CDF& cdf, int distribution, int n, double u) {
    const double total = cdf(distribution, n - 1);
    if (!(total > 0.0)) return 0; // Error is reported at the sweep boundary.
    double target = u * total;
    if (!(target < total)) target = Kokkos::nextafter(total, 0.0);
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (cdf(distribution, mid) > target) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

template<class ExecutionSpace>
class AncestorSampler {
public:
    using MemorySpace = typename ExecutionSpace::memory_space;
    using Vector = Kokkos::View<double*, MemorySpace>;
    using Indices = Kokkos::View<int*, MemorySpace>;
    using CDF = Kokkos::View<double**, MemorySpace>;
    using Error = Kokkos::View<int, MemorySpace>;

    explicit AncestorSampler(int n) : n_(n), cdf_("ancestor_cdf", 2, n),
                                      error_("ancestor_error") {
        if (n < 1) throw std::invalid_argument("Ancestor sampler requires N >= 1");
    }

    void build(const Vector& ordinary, const Vector& conditional) {
        const int n = n_;
        const auto cdf = cdf_;
        const auto error = error_;
        using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
        using Member = typename Policy::member_type;
        Kokkos::parallel_for("ancestor_cdf", Policy(2, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Member& team) {
                const int d = team.league_rank();
                const auto logs = d == 0 ? ordinary : conditional;
                double maximum;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n),
                    [&](int j, double& m) { if (logs(j) > m) m = logs(j); },
                    Kokkos::Max<double>(maximum));
                int invalid;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n),
                    [&](int j, int& count) {
                        if (Kokkos::isnan(logs(j)) || logs(j) == INFINITY) ++count;
                    }, invalid);
                // Kokkos::Max initializes to lowest finite value, not -infinity.
                int finite;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n),
                    [&](int j, int& count) { if (Kokkos::isfinite(logs(j))) ++count; }, finite);
                const bool valid = invalid == 0 && finite > 0;
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    if (!valid) Kokkos::atomic_fetch_or(&error(), 1 << d);
                });
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, n),
                    [&](int j, double& sum, bool final) {
                        sum += valid ? Kokkos::exp(logs(j) - maximum) : 0.0;
                        if (final) cdf(d, j) = sum;
                    });
            });
    }

    void draw(const Indices& out, uint64_t seed, uint64_t sweep, uint32_t time) const {
        const auto cdf = cdf_;
        const int n = n_;
        Kokkos::parallel_for("ancestor_draw", Kokkos::RangePolicy<ExecutionSpace>(0, n),
            KOKKOS_LAMBDA(int j) {
                out(j) = invert_cdf(cdf, j == 0 ? 1 : 0, n,
                                    ancestor_uniform(seed, sweep, time, j));
            });
    }

    void check() const {
        int error = 0;
        Kokkos::deep_copy(error, error_);
        if (error) throw std::runtime_error(
            "Invalid ancestor weights (NaN, +infinity or zero total mass), distribution mask="
            + std::to_string(error));
    }

    const CDF& cdf() const { return cdf_; }

private:
    int n_;
    CDF cdf_;
    Error error_;
};
} // namespace pgas
#endif
