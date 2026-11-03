# Run before project(): CMake needs the vcpkg toolchain before compiler setup.
# An explicitly supplied toolchain keeps control of dependency provisioning.
if(CMAKE_TOOLCHAIN_FILE AND NOT CSP_MANAGED_TOOLCHAIN)
    return()
endif()
if(NOT CMAKE_HOST_SYSTEM_NAME MATCHES "^(Linux|Windows|Darwin)$")
    message(FATAL_ERROR "Automatic dependency setup supports Linux/Windows x86-64 and macOS arm64")
endif()

set(CSP_VCPKG_ROOT "${CMAKE_SOURCE_DIR}/build/tools/vcpkg" CACHE PATH "Pinned vcpkg checkout")
set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/build/vcpkg-installed" CACHE PATH "Native dependencies")
# scikit-build-core supplies its build interpreter. Do not run FindPython
# before project(): platform suffixes are not initialized yet on Windows.
if(NOT Python_EXECUTABLE)
    find_program(Python_EXECUTABLE NAMES python3 python REQUIRED)
endif()
message(STATUS "Preparing pinned C-SPIKES native dependencies (cached under build/)")
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/build_native.py"
        --dependencies-only --vcpkg-root "${CSP_VCPKG_ROOT}"
        --dependency-root "${VCPKG_INSTALLED_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(_csp_triplet arm64-osx-csp)
    set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "Apple Silicon architecture")
    set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0 CACHE STRING "Minimum macOS version")
    if(NOT CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
        find_program(_csp_gfortran NAMES gfortran-14 gfortran
            HINTS /opt/homebrew/opt/gcc@14/bin REQUIRED)
        set(CMAKE_Fortran_COMPILER "${_csp_gfortran}" CACHE FILEPATH "Fortran compiler")
    endif()
else()
    string(TOLOWER "${CMAKE_HOST_SYSTEM_NAME}" _csp_host)
    set(_csp_triplet "x64-${_csp_host}-csp")
endif()
set(VCPKG_TARGET_TRIPLET "${_csp_triplet}" CACHE STRING "Pinned native dependency triplet")
set(VCPKG_MANIFEST_MODE OFF CACHE BOOL "Dependencies were prepared above")
set(CMAKE_TOOLCHAIN_FILE "${CSP_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    CACHE FILEPATH "Pinned vcpkg toolchain")
set(CSP_MANAGED_TOOLCHAIN ON CACHE INTERNAL "C-SPIKES prepared this toolchain")
