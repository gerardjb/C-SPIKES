cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX Fortran)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

add_library(pgas_obj_cpu OBJECT ${PGAS_SRCS})
target_compile_definitions(pgas_obj_cpu PRIVATE USE_GPU=0)
target_compile_options(pgas_obj_cpu PRIVATE -Wno-error=cpp)
target_include_directories(pgas_obj_cpu PRIVATE src/spike_find/pgas)
target_link_libraries(pgas_obj_cpu PRIVATE ${PGAS_COMMON_LIBS})

pybind11_add_module(pgas_bound_cpu MODULE src/c_spikes/pgas/bindings.cpp $<TARGET_OBJECTS:pgas_obj_cpu> NO_EXTRAS)
target_compile_options(pgas_bound_cpu PRIVATE -Wno-error=cpp)
target_compile_definitions(pgas_bound_cpu PRIVATE PGAS_PYBIND_MODULE=pgas_bound_cpu)
target_include_directories(pgas_bound_cpu PRIVATE src/spike_find/pgas)
target_link_libraries(pgas_bound_cpu PRIVATE ${PGAS_COMMON_LIBS})

if(_pgas_build_gpu)
    add_library(pgas_obj_gpu OBJECT ${PGAS_SRCS})
    target_compile_definitions(pgas_obj_gpu PRIVATE USE_GPU=1)
    target_compile_options(pgas_obj_gpu PRIVATE -Wno-error=cpp)
    target_include_directories(pgas_obj_gpu PRIVATE src/spike_find/pgas)
    target_link_libraries(pgas_obj_gpu PRIVATE ${PGAS_COMMON_LIBS})

    pybind11_add_module(pgas_bound_gpu MODULE src/c_spikes/pgas/bindings.cpp $<TARGET_OBJECTS:pgas_obj_gpu> NO_EXTRAS)
    target_compile_options(pgas_bound_gpu PRIVATE -Wno-error=cpp)
    target_compile_definitions(pgas_bound_gpu PRIVATE PGAS_PYBIND_MODULE=pgas_bound_gpu)
    target_include_directories(pgas_bound_gpu PRIVATE src/spike_find/pgas)
    target_link_libraries(pgas_bound_gpu PRIVATE ${PGAS_COMMON_LIBS})
endif()

install(TARGETS pgas_bound_cpu DESTINATION ${SKBUILD_PROJECT_NAME}/pgas)
if(_pgas_build_gpu)
    install(TARGETS pgas_bound_gpu DESTINATION ${SKBUILD_PROJECT_NAME}/pgas)
endif()
