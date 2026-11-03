# Kokkos supports one compile-time GPU feature baseline, not multiple
# Kokkos_ARCH_* options. Keep the Turing baseline for all device compilation,
# and ask ptxas for additional native images of that same compute_75 program.
# CMAKE_CUDA_ARCHITECTURES=75 supplies both sm_75 and forward-compatible PTX.
# Apply these options before FetchContent so Kokkos and PGAS carry the same
# images. No upstream Kokkos patch or architecture-specific PTX is needed.
option(CSP_CUDA_PORTABLE_WHEEL "Build Turing-baseline CUDA code for multiple GPU generations" OFF)
if(CSP_CUDA_PORTABLE_WHEEL)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.9)
        message(FATAL_ERROR "Portable CUDA wheels require CUDA 12.9 or newer")
    endif()
    if(NOT Kokkos_ARCH_TURING75 OR NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
        message(FATAL_ERROR "Portable CUDA wheels require Kokkos_ARCH_TURING75=ON and CMAKE_CUDA_ARCHITECTURES=75")
    endif()
    foreach(arch 80 86 89 90 100 103 120 121)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=sm_${arch}>")
    endforeach()
endif()
