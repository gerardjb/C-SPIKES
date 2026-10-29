#ifndef CSP_NUMERICAL_DIAGNOSTICS_H
#define CSP_NUMERICAL_DIAGNOSTICS_H

// One build-wide choice; standalone native clients default to the ordinary mode.
#ifndef C_SPIKES_OBSERVATIONAL_DIAGNOSTICS
#define C_SPIKES_OBSERVATIONAL_DIAGNOSTICS 0
#endif
#if C_SPIKES_OBSERVATIONAL_DIAGNOSTICS != 0 && C_SPIKES_OBSERVATIONAL_DIAGNOSTICS != 1
#error "C_SPIKES_OBSERVATIONAL_DIAGNOSTICS must be 0 or 1"
#endif
// Elide observation arguments as well as scans. Never wrap required validation.
#if C_SPIKES_OBSERVATIONAL_DIAGNOSTICS
#define CSP_OBSERVE(...) __VA_ARGS__
#else
#define CSP_OBSERVE(...)
#endif

// Observation only. No model storage, numerical decisions, or device operations.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace csp_diagnostics {
inline constexpr bool enabled = C_SPIKES_OBSERVATIONAL_DIAGNOSTICS != 0;
enum Category { state_value, rate_value, output_value, normalization, categories };
struct Location {
    const char* method = "unknown";
    const char* field = "unknown";
    const char* integrator = "unknown";
    int mode = -1;
    std::int64_t substep = -1;
    std::int64_t index = -1;
    double requested_dt = 0, h = 0, value = 0;
    bool timing_known = false, calcium_known = false;
    double ca = 0, bca = 0, ca_in = 0;
};
struct Record {
    const char* operation = "unknown";
    std::uint64_t id = 0, detections = 0;
    std::uint64_t counts[categories] = {};
    Location current, first;
    bool complete = true;
};
using Sink = void (*)(const Record&);
struct Active { Record record; Sink sink = nullptr; };
inline Active*& active() noexcept { static thread_local Active* p = nullptr; return p; }
inline bool& emitting() noexcept { static thread_local bool value = false; return value; }
inline std::uint64_t next_id() noexcept {
    static std::atomic<std::uint64_t> value{0};
    return value.fetch_add(1, std::memory_order_relaxed);
}
inline void increment(std::uint64_t& n) noexcept {
    if (n != std::numeric_limits<std::uint64_t>::max()) ++n;
}
inline const char* classification(double x) noexcept {
    return std::isnan(x) ? "NaN" : (std::isinf(x) ? (x > 0 ? "+Inf" : "-Inf") : "zero");
}
inline void format(const Record& r, char* first, std::size_t nf,
                   char* summary, std::size_t ns, char* debug, std::size_t nd) noexcept {
    const auto& f = r.first;
#if defined(__VERSION__)
    const char* compiler = __VERSION__;
#else
    const char* compiler = "unknown";
#endif
    std::snprintf(first, nf,
        "Numerical diagnostic: %s observed; operation=%s id=%llu method=%s field=%s index=%lld "
        "mode=%d integrator=%s requested_dt=%.17g nominal_h=%.17g substep=%lld timing_known=%d. "
        "Execution continues with existing behavior; cause has not been established.",
        classification(f.value), r.operation, static_cast<unsigned long long>(r.id),
        f.method, f.field, static_cast<long long>(f.index), f.mode, f.integrator,
        f.requested_dt, f.h, static_cast<long long>(f.substep), int(f.timing_known));
    std::snprintf(summary, ns,
        "Numerical diagnostic summary: operation=%s id=%llu detections=%llu "
        "state=%llu rates=%llu output=%llu normalization=%llu suppressed=%llu status=%s. "
        "Counts are observations, not independent failures; first detected location is in the preceding record.",
        r.operation, static_cast<unsigned long long>(r.id),
        static_cast<unsigned long long>(r.detections),
        static_cast<unsigned long long>(r.counts[state_value]),
        static_cast<unsigned long long>(r.counts[rate_value]),
        static_cast<unsigned long long>(r.counts[output_value]),
        static_cast<unsigned long long>(r.counts[normalization]),
        static_cast<unsigned long long>(r.detections ? r.detections - 1 : 0),
        r.complete ? "complete" : "interrupted");
    std::snprintf(debug, nd,
        "Numerical diagnostic context: id=%llu calcium_known=%d Ca=%.17g BCa=%.17g Ca_in=%.17g "
        "backend=host compiler=%s. Version/source revision and seed are not captured by this collector.",
        static_cast<unsigned long long>(r.id), int(f.calcium_known), f.ca, f.bca, f.ca_in, compiler);
}
inline void stderr_sink(const Record& r) noexcept {
    char first[1024], summary[768], debug[512];
    format(r, first, sizeof(first), summary, sizeof(summary), debug, sizeof(debug));
    std::fprintf(stderr, "WARNING %s\nWARNING %s\n", first, summary);
    const char* flag = std::getenv("C_SPIKES_NUMERICAL_DEBUG");
    if (flag && flag[0] == '1' && flag[1] == '\0') std::fprintf(stderr, "DEBUG %s\n", debug);
}

#if C_SPIKES_OBSERVATIONAL_DIAGNOSTICS
class Scope {
    Active local_;
    Active* target_ = nullptr;
    Location previous_;
    const char* method_;
    int exceptions_;
    bool owner_ = false;
public:
    explicit Scope(const char* method, Sink sink = nullptr) noexcept
        : method_(method), exceptions_(std::uncaught_exceptions()) {
        if (emitting()) return;
        target_ = active();
        if (!target_) {
            // Parallel-host and device aggregation is a separate milestone.
            // In particular, do not emit one stderr report per OpenMP particle.
#ifdef _OPENMP
            if (omp_in_parallel()) return;
#endif
            owner_ = true;
            target_ = &local_;
            target_->record.operation = method;
            target_->record.id = next_id();
            target_->sink = sink;
            active() = target_;
        } else if (target_->record.operation == python_operation()) {
            target_->record.operation = method;
        }
        previous_ = target_->record.current;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    static const char* python_operation() noexcept { static const char name[] = "Python GCaMP call"; return name; }
    ~Scope() noexcept {
        if (target_) target_->record.current = previous_;
        if (!owner_) return;
        active() = nullptr;
        if (!target_->record.detections) return;
        target_->record.complete = std::uncaught_exceptions() == exceptions_;
        emitting() = true;
        try { (target_->sink ? target_->sink : stderr_sink)(target_->record); } catch (...) {}
        emitting() = false;
    }
    void mode(int value) noexcept { if (target_) target_->record.current.mode = value; }
    // Matrix/equilibrium construction does not select an evolution scheme.
    // Scope destruction restores the enclosing location, including its timing.
    void equilibrium_context() noexcept {
        if (!target_) return;
        auto& c = target_->record.current;
        c.mode = -1; c.integrator = "not_applicable";
        c.requested_dt = 0; c.h = 0; c.substep = -1; c.timing_known = false;
    }
    void step(int mode, const char* integrator, double requested_dt, double h,
              std::int64_t substep, double ca, double bca, double ca_in) noexcept {
        if (!target_) return;
        auto& c = target_->record.current;
        c.mode = mode; c.integrator = integrator; c.requested_dt = requested_dt;
        c.h = h; c.substep = substep; c.timing_known = true;
        c.ca = ca; c.bca = bca; c.ca_in = ca_in; c.calcium_known = true;
    }
    void scalar(double value, const char* field, Category category,
                std::int64_t index = -1, bool zero_invalid = false) noexcept {
        if (!target_ || (std::isfinite(value) && !(zero_invalid && value == 0))) return;
        auto& r = target_->record;
        if (!r.detections) {
            r.first = r.current; r.first.method = method_; r.first.field = field;
            r.first.index = index; r.first.value = value;
        }
        increment(r.detections); increment(r.counts[category]);
    }
    template<class Array> void array(const Array& a, const char* field, Category category) noexcept {
        if (!target_) return;
        // One detection per array observation; record its first nonfinite index.
        for (std::size_t i = 0; i < a.n_elem; ++i)
            if (!std::isfinite(a[i])) { scalar(a[i], field, category, i); break; }
    }
};
#else
// Empty call guard for clients sharing this header; production observations use
// CSP_OBSERVE so even their arguments are absent in an OFF build.
class Scope {
public:
    explicit Scope(const char*, Sink = nullptr) noexcept {}
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void mode(int) noexcept {}
    void equilibrium_context() noexcept {}
    void step(int, const char*, double, double, std::int64_t, double, double, double) noexcept {}
    void scalar(double, const char*, Category, std::int64_t = -1, bool = false) noexcept {}
    template<class Array> void array(const Array&, const char*, Category) noexcept {}
};
#endif
} // namespace csp_diagnostics
#endif
