# Portable build profile. This is distinct from the historical hash-qualified
# provider; passing build/smoke tests is not historical scientific acceptance.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "portable-lp64 requires a 64-bit build")
endif()
find_package(OpenBLAS CONFIG REQUIRED COMPONENTS serial)
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES OpenBLAS::OpenBLAS)
check_cxx_source_compiles("#include <openblas_config.h>
static_assert(sizeof(blasint) == 4, \"C-SPIKES requires LP64 BLAS integers\");
int main() { return 0; }" CSP_OPENBLAS_IS_LP64)
unset(CMAKE_REQUIRED_LIBRARIES)
if(NOT CSP_OPENBLAS_IS_LP64)
    message(FATAL_ERROR "portable-lp64 requires an OpenBLAS build with 32-bit BLAS integers")
endif()
if(NOT WIN32)
    enable_language(Fortran)
endif()
find_package(LAPACK REQUIRED)
add_library(csp_numerical_provider INTERFACE)
if(WIN32)
    target_link_libraries(csp_numerical_provider INTERFACE OpenBLAS::OpenBLAS LAPACK::LAPACK)
else()
    # LAPACK references BLAS; BLAS can reference LAPACK error handlers. Keep
    # both archives in a rescan group rather than relying on duplicate entries
    # that CMake may deduplicate through imported targets.
    set(_csp_lapack ${LAPACK_LIBRARIES})
    list(REMOVE_ITEM _csp_lapack "${OpenBLAS_LIBRARY}")
    list(JOIN _csp_lapack "," _csp_lapack_group)
    target_link_libraries(csp_numerical_provider INTERFACE
        "$<LINK_GROUP:RESCAN,OpenBLAS::OpenBLAS,${_csp_lapack_group}>")
    target_link_libraries(csp_numerical_provider INTERFACE ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
    target_link_directories(csp_numerical_provider INTERFACE ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
endif()
include("${CMAKE_CURRENT_LIST_DIR}/PythonModuleExports.cmake")
