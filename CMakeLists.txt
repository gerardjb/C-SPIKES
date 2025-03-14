cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB PGAS_SRCS CONFIGURE_DEPENDS src/spike_find/pgas/*.cpp)

pybind11_add_module(pgas_bound MODULE src/spike_find/pgas/bindings.cpp ${PGAS_SRCS})

target_include_directories(pgas_bound PRIVATE src/spike_find/pgas)

target_link_libraries(pgas_bound PRIVATE armadillo JsonCpp::JsonCpp GSL::gsl GSL::gslcblas)
target_link_libraries(pgas_bound PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS pgas_bound DESTINATION ${SKBUILD_PROJECT_NAME}/pgas)
