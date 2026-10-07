#ifndef C_SPIKES_RESAMPLING_OPTIONS_H
#define C_SPIKES_RESAMPLING_OPTIONS_H

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace pgas {
struct ResamplingOptions {
    bool device = false;
    bool selected_trajectory = false;
    uint64_t seed = 0;

    static ResamplingOptions from_environment(unsigned long effective_seed) {
        ResamplingOptions out;
        out.seed = effective_seed;
        const char* mode = std::getenv("C_SPIKES_PGAS_RESAMPLING");
        if (mode && std::string(mode) != "host") {
            if (std::string(mode) != "device")
                throw std::invalid_argument("C_SPIKES_PGAS_RESAMPLING must be host or device");
            out.device = true;
        }
        const char* trajectory = std::getenv("C_SPIKES_PGAS_TRAJECTORY");
        if (trajectory && std::string(trajectory) != "full") {
            if (std::string(trajectory) != "selected")
                throw std::invalid_argument("C_SPIKES_PGAS_TRAJECTORY must be full or selected");
            out.selected_trajectory = true;
        }
#if !defined(USE_GPU) || !USE_GPU
        if (out.selected_trajectory)
            throw std::invalid_argument("Selected trajectory extraction requires the GPU backend");
        if (out.device) throw std::invalid_argument("Device ancestor sampling requires the GPU backend");
#endif
        const char* seed = std::getenv("C_SPIKES_PGAS_ANCESTOR_SEED");
        if (seed) {
            if (!out.device) throw std::invalid_argument("Ancestor seed requires device resampling");
            const std::string value(seed);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("Ancestor seed must be an unsigned 64-bit decimal integer");
            out.seed = std::stoull(value);
        }
        return out;
    }
};
} // namespace pgas
#endif
