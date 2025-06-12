cmake_minimum_required(VERSION 3.15...3.26)

# I added Fortran here. For some reason, fortran in armadillo wasn't linking
# properly without this. I don't know why.
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX Fortran)

find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB PGAS_SRCS CONFIGURE_DEPENDS src/c_spikes/pgas/*.cpp)

# Tell CMake where vcpkg lives
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg" CACHE STRING "")

# Force both build- and install-RPATH to point at your vcpkg libs dir
set(_vpkglib "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-linux/lib")
set(CMAKE_BUILD_RPATH "${_vpkglib}")
set(CMAKE_INSTALL_RPATH "${_vpkglib}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

pybind11_add_module(pgas_bound MODULE src/c_spikes/pgas/bindings.cpp ${PGAS_SRCS})

target_include_directories(pgas_bound PRIVATE src/c_spikes/pgas)

target_link_libraries(pgas_bound PRIVATE armadillo JsonCpp::JsonCpp GSL::gsl GSL::gslcblas)
target_link_libraries(pgas_bound PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS pgas_bound DESTINATION ${SKBUILD_PROJECT_NAME}/pgas)
