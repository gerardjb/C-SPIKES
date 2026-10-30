# Kokkos needs the OpenMP 3+ API, absent from MSVC's default vcomp runtime.
# Use MSVC's LLVM code generation with our source-built, redistributable libomp.
set(_csp_openmp_root "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
find_library(_csp_openmp_library NAMES libomp
    PATHS "${_csp_openmp_root}/lib" NO_DEFAULT_PATH REQUIRED)
foreach(language C CXX)
    set(OpenMP_${language}_FLAGS "/openmp:llvm" CACHE STRING "MSVC LLVM OpenMP flags" FORCE)
    set(OpenMP_${language}_LIB_NAMES "libomp" CACHE STRING "Pinned LLVM OpenMP runtime" FORCE)
    set(OpenMP_${language}_INCLUDE_DIR "${_csp_openmp_root}/include" CACHE PATH "LLVM OpenMP headers" FORCE)
endforeach()
set(OpenMP_libomp_LIBRARY "${_csp_openmp_library}" CACHE FILEPATH "Pinned LLVM OpenMP import library" FORCE)

# MSVC keeps _OPENMP at 200203 with /openmp:llvm. FindOpenMP uses that
# macro for its version, although LLVM supplies the newer runtime API that
# Kokkos 5 requires. Verify the API and OpenMP linkage before giving CMake
# the 3.0 version hint; do not redefine _OPENMP or patch Kokkos's checks.
include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "/openmp:llvm")
set(CMAKE_REQUIRED_INCLUDES "${_csp_openmp_root}/include")
set(CMAKE_REQUIRED_LIBRARIES "${_csp_openmp_library}")
check_cxx_source_compiles([=[
#include <omp.h>
int main() {
  omp_set_max_active_levels(2);
  int levels = 0;
  #pragma omp parallel reduction(+:levels)
  {
    levels += omp_get_level() + omp_get_active_level();
    levels += omp_get_ancestor_thread_num(0) + omp_get_team_size(0);
  }
  return levels + omp_get_max_active_levels();
}
]=] CSP_LLVM_OPENMP_30_API)
cmake_pop_check_state()
if(NOT CSP_LLVM_OPENMP_30_API)
    message(FATAL_ERROR "MSVC /openmp:llvm and the pinned runtime must provide the OpenMP 3.0 API required by Kokkos")
endif()
set(OpenMP_CXX_SPEC_DATE "200805" CACHE STRING "Verified LLVM OpenMP 3.0 runtime API with MSVC" FORCE)
