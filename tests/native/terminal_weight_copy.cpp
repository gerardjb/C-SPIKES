#include "terminal_weight_copy.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

#if defined(CSP_COPY_TEST_CUDA)
using CopyExec = Kokkos::Cuda;
using CopyMemory = Kokkos::CudaSpace;
#else
using CopyExec = Kokkos::OpenMP;
using CopyMemory = Kokkos::HostSpace;
#endif

static void exact(double actual, double expected) {
    if (std::memcmp(&actual, &expected, sizeof(double)) != 0)
        throw std::runtime_error("terminal copy bitwise mismatch");
}

static void check() {
    using History = Kokkos::View<double**, CopyMemory>;
    using Terminal = Kokkos::View<double*, CopyMemory>;
    History history("copy_history", 9, 4);
    Terminal terminal("copy_terminal", 9);
    auto initial = Kokkos::create_mirror(history);
    auto observed = Kokkos::create_mirror(history);
    auto host_terminal = Kokkos::create_mirror(terminal);
    for (int j = 0; j < 9; ++j)
        for (unsigned int t = 0; t < 4; ++t)
            initial(j, t) = (j == 0 && t == 2) ? -0.0 : 100.0 * j + t + 0.25;
    Kokkos::deep_copy(history, initial);
    for (unsigned int last : {0u, 2u, 3u}) {
        for (int count : {0, 1, 7}) {
            Kokkos::deep_copy(terminal, -777.0);
            Kokkos::parallel_for("terminal_weights", Kokkos::RangePolicy<CopyExec>(0, count),
                csp_pgas_detail::TerminalWeightCopy<History, Terminal>{history, terminal, last});
            // Match production's host-visibility boundary; no extra fence.
            Kokkos::deep_copy(host_terminal, terminal);
            for (int j = 0; j < 9; ++j)
                exact(host_terminal(j), j < count ? initial(j, last) : -777.0);
            Kokkos::deep_copy(observed, history);
            for (int j = 0; j < 9; ++j)
                for (unsigned int t = 0; t < 4; ++t)
                    exact(observed(j, t), initial(j, t));
        }
    }
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int status = 0;
    try {
        check();
        std::cout << "terminal_weight_copy: PASS (9 cases)\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    Kokkos::finalize();
    return status;
}
