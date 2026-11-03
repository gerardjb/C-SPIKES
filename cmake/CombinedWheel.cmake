# Kokkos has one backend configuration per CMake build tree. Compile the CPU
# and CUDA extensions in separate trees so importing CPU never needs a driver.
# The ordinary project remains the implementation of both child builds.
include(ExternalProject)
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(_csp_common
    -DCSP_COMBINED_WHEEL:BOOL=OFF
    -DCSP_AUTO_DEPENDENCIES:BOOL=OFF
    -DPython_EXECUTABLE:FILEPATH=${Python_EXECUTABLE}
    -Dpybind11_DIR:PATH=${pybind11_DIR})
foreach(variable CMAKE_BUILD_TYPE CMAKE_C_COMPILER CMAKE_CXX_COMPILER
        CMAKE_TOOLCHAIN_FILE CMAKE_MAKE_PROGRAM CMAKE_PREFIX_PATH
        VCPKG_INSTALLED_DIR VCPKG_TARGET_TRIPLET VCPKG_MANIFEST_MODE
        SKBUILD_PROJECT_NAME SKBUILD_PROJECT_VERSION SKBUILD_STATE
        CSP_NUMERICAL_PROFILE C_SPIKES_TEST_RESAMPLING
        C_SPIKES_OBSERVATIONAL_DIAGNOSTICS)
    if(DEFINED ${variable})
        string(REPLACE ";" "|" value "${${variable}}")
        list(APPEND _csp_common "-D${variable}:STRING=${value}")
    endif()
endforeach()
set(_csp_cpu_build "${CMAKE_CURRENT_BINARY_DIR}/cpu")
set(_csp_gpu_build "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(_csp_cpu_install "${CMAKE_CURRENT_BINARY_DIR}/stage-cpu")
set(_csp_gpu_install "${CMAKE_CURRENT_BINARY_DIR}/stage-cuda")

ExternalProject_Add(csp_cpu_build
    SOURCE_DIR "${CMAKE_SOURCE_DIR}"
    BINARY_DIR "${_csp_cpu_build}"
    INSTALL_DIR "${_csp_cpu_install}"
    LIST_SEPARATOR "|"
    CMAKE_ARGS ${_csp_common}
        -DPGAS_BUILD_GPU:STRING=OFF -DKokkos_ENABLE_CUDA:BOOL=OFF
        -DCSP_CUDA_PORTABLE_WHEEL:BOOL=OFF
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
        --config $<CONFIG> --component CSPPython --prefix <INSTALL_DIR>)

set(_csp_cuda_args)
get_cmake_property(_csp_variables VARIABLES)
foreach(variable IN LISTS _csp_variables)
    if(variable MATCHES "^Kokkos_ARCH_" OR variable MATCHES "^CMAKE_CUDA_(COMPILER|HOST_COMPILER|ARCHITECTURES)$"
            OR variable STREQUAL "CSP_CUDA_PORTABLE_WHEEL")
        string(REPLACE ";" "|" value "${${variable}}")
        list(APPEND _csp_cuda_args "-D${variable}:STRING=${value}")
    endif()
endforeach()
ExternalProject_Add(csp_cuda_build
    DEPENDS csp_cpu_build
    SOURCE_DIR "${CMAKE_SOURCE_DIR}"
    BINARY_DIR "${_csp_gpu_build}"
    INSTALL_DIR "${_csp_gpu_install}"
    LIST_SEPARATOR "|"
    CMAKE_ARGS ${_csp_common} ${_csp_cuda_args}
        -DPGAS_BUILD_GPU:STRING=ON -DKokkos_ENABLE_CUDA:BOOL=ON
        -DFETCHCONTENT_SOURCE_DIR_KOKKOS:PATH=${_csp_cpu_build}/_deps/kokkos-src
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
        --config $<CONFIG> --component CSPPython --prefix <INSTALL_DIR>)

# Use the independent CPU module, OASIS and common notices from the CPU build.
# Only the GPU extension and CUDA notices are taken from the CUDA build.
install(DIRECTORY "${_csp_cpu_install}/" DESTINATION . COMPONENT CSPPython)
install(DIRECTORY "${_csp_gpu_install}/${SKBUILD_PROJECT_NAME}/pgas/"
    DESTINATION "${SKBUILD_PROJECT_NAME}/pgas" COMPONENT CSPPython
    FILES_MATCHING PATTERN "pgas_bound_gpu.*")
install(DIRECTORY "${_csp_gpu_install}/c_spikes/licenses/cuda"
    DESTINATION c_spikes/licenses COMPONENT CSPPython)

if(C_SPIKES_TEST_RESAMPLING)
    enable_testing()
    add_test(NAME independent_cpu_checks COMMAND ${CMAKE_CTEST_COMMAND}
        --test-dir "${_csp_cpu_build}" -C $<CONFIG> --output-on-failure --timeout 120)
    add_test(NAME cuda_checks COMMAND ${CMAKE_CTEST_COMMAND}
        --test-dir "${_csp_gpu_build}" -C $<CONFIG> -R "_gpu$" --output-on-failure --timeout 120)
endif()
