cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CSPIKES_ENABLE_GPU "OFF" CACHE STRING "Enable CUDA backend for Kokkos/PGAS (AUTO, ON, OFF).")
set_property(CACHE CSPIKES_ENABLE_GPU PROPERTY STRINGS AUTO ON OFF)

if(NOT CSPIKES_ENABLE_GPU STREQUAL "AUTO"
   AND NOT CSPIKES_ENABLE_GPU STREQUAL "ON"
   AND NOT CSPIKES_ENABLE_GPU STREQUAL "OFF")
    message(FATAL_ERROR "CSPIKES_ENABLE_GPU must be one of: AUTO, ON, OFF.")
endif()

# Default to CPU-only unless explicitly requested; callers can still pass
# -DKokkos_ENABLE_CUDA=ON directly.
if(NOT DEFINED Kokkos_ENABLE_CUDA)
    if(CSPIKES_ENABLE_GPU STREQUAL "ON")
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend.")
    else()
        set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Enable Kokkos CUDA backend.")
    endif()
endif()

add_subdirectory(dependencies)

find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED)

file(GLOB PGAS_SRCS CONFIGURE_DEPENDS src/c_spikes/pgas/*.cpp)
list(REMOVE_ITEM PGAS_SRCS "${CMAKE_SOURCE_DIR}/src/c_spikes/pgas/bindings.cpp")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo") # for debugging CUDA code

# Tell CMake where vcpkg lives
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg" CACHE STRING "")

# Force both build- and install-RPATH to point at your vcpkg libs dir
set(_vpkglib "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-linux/lib")
set(CMAKE_BUILD_RPATH "${_vpkglib}")
set(CMAKE_INSTALL_RPATH "${_vpkglib}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(PGAS_BUILD_GPU "AUTO" CACHE STRING "Build PGAS GPU backend (AUTO, ON, OFF).")
set_property(CACHE PGAS_BUILD_GPU PROPERTY STRINGS AUTO ON OFF)

set(_pgas_build_gpu OFF)
if(PGAS_BUILD_GPU STREQUAL "ON")
    if(NOT Kokkos_ENABLE_CUDA)
        message(FATAL_ERROR "PGAS_BUILD_GPU=ON but Kokkos CUDA backend is not enabled.")
    endif()
    set(_pgas_build_gpu ON)
elseif(PGAS_BUILD_GPU STREQUAL "OFF")
    set(_pgas_build_gpu OFF)
else()
    if(Kokkos_ENABLE_CUDA)
        set(_pgas_build_gpu ON)
    endif()
endif()

set(PGAS_COMMON_LIBS armadillo JsonCpp::JsonCpp GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX Kokkos::kokkos)

set(PGAS_COMPILER_OPTIONS "")
if(MSVC)
    list(APPEND PGAS_COMPILER_OPTIONS /EHsc)
else()
    list(APPEND PGAS_COMPILER_OPTIONS -Wno-error=cpp)
endif()

add_library(pgas_obj_cpu OBJECT ${PGAS_SRCS})
target_compile_definitions(pgas_obj_cpu PRIVATE USE_GPU=0)
target_compile_options(pgas_obj_cpu PRIVATE ${PGAS_COMPILER_OPTIONS})
target_include_directories(pgas_obj_cpu PRIVATE src/c_spikes/pgas)
target_link_libraries(pgas_obj_cpu PRIVATE ${PGAS_COMMON_LIBS})

pybind11_add_module(pgas_bound_cpu MODULE src/c_spikes/pgas/bindings.cpp $<TARGET_OBJECTS:pgas_obj_cpu> NO_EXTRAS)
target_compile_options(pgas_bound_cpu PRIVATE ${PGAS_COMPILER_OPTIONS})
target_compile_definitions(pgas_bound_cpu PRIVATE PGAS_PYBIND_MODULE=pgas_bound_cpu)
target_include_directories(pgas_bound_cpu PRIVATE src/c_spikes/pgas)
target_link_libraries(pgas_bound_cpu PRIVATE ${PGAS_COMMON_LIBS})

if(_pgas_build_gpu)
    add_library(pgas_obj_gpu OBJECT ${PGAS_SRCS})
    target_compile_definitions(pgas_obj_gpu PRIVATE USE_GPU=1)
    target_compile_options(pgas_obj_gpu PRIVATE ${PGAS_COMPILER_OPTIONS})
    target_include_directories(pgas_obj_gpu PRIVATE src/c_spikes/pgas)
    target_link_libraries(pgas_obj_gpu PRIVATE ${PGAS_COMMON_LIBS})

    pybind11_add_module(pgas_bound_gpu MODULE src/c_spikes/pgas/bindings.cpp $<TARGET_OBJECTS:pgas_obj_gpu> NO_EXTRAS)
    target_compile_options(pgas_bound_gpu PRIVATE ${PGAS_COMPILER_OPTIONS})
    target_compile_definitions(pgas_bound_gpu PRIVATE PGAS_PYBIND_MODULE=pgas_bound_gpu)
    target_include_directories(pgas_bound_gpu PRIVATE src/c_spikes/pgas)
    target_link_libraries(pgas_bound_gpu PRIVATE ${PGAS_COMMON_LIBS})
endif()

install(TARGETS pgas_bound_cpu DESTINATION ${SKBUILD_PROJECT_NAME}/pgas)
if(_pgas_build_gpu)
    install(TARGETS pgas_bound_gpu DESTINATION ${SKBUILD_PROJECT_NAME}/pgas)
endif()

if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
    set(_vcpkg_bin_dir "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin")
    set(_runtime_modules "\"$<TARGET_FILE:pgas_bound_cpu>\"")
    if(_pgas_build_gpu)
        string(APPEND _runtime_modules "\n    \"$<TARGET_FILE:pgas_bound_gpu>\"")
    endif()

    install(CODE
"set(_pgas_dest \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${SKBUILD_PROJECT_NAME}/pgas\")
file(MAKE_DIRECTORY \"\${_pgas_dest}\")
file(GET_RUNTIME_DEPENDENCIES
  RESOLVED_DEPENDENCIES_VAR _resolved
  UNRESOLVED_DEPENDENCIES_VAR _unresolved
  MODULES
    ${_runtime_modules}
  DIRECTORIES
    \"${_vcpkg_bin_dir}\"
  PRE_EXCLUDE_REGEXES
    \"api-ms-win-.*\"
    \"ext-ms-.*\"
    \"python[0-9]+\\\\.dll\"
    \"AzureAttest.*\\\\.dll\"
    \"HvsiFileTrust\\\\.dll\"
    \"PdmUtilities\\\\.dll\"
    \"wpaxholder\\\\.dll\"
  POST_EXCLUDE_REGEXES
    \".*/[Ww][Ii][Nn][Dd][Oo][Ww][Ss]/.*\"
    \".*/[Pp]rogram [Ff]iles( \\(x86\\))?/.*\"
)
if(_unresolved)
  message(STATUS \"Unresolved runtime dependencies (ignored): \${_unresolved}\")
endif()
list(REMOVE_DUPLICATES _resolved)
foreach(_dep IN LISTS _resolved)
  file(INSTALL DESTINATION \"\${_pgas_dest}\" TYPE SHARED_LIBRARY FILES \"\${_dep}\" FOLLOW_SYMLINK_CHAIN)
endforeach()
")
endif()
