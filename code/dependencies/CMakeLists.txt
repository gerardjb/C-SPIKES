include(FetchContent)

if(DEFINED ENV{KOKKOS_SOURCE_DIR} AND EXISTS "$ENV{KOKKOS_SOURCE_DIR}/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_KOKKOS "$ENV{KOKKOS_SOURCE_DIR}" CACHE PATH "Pre-fetched Kokkos source directory")
endif()

set(_pgas_default_kokkos_arch "TURING75")
if(DEFINED ENV{PGAS_KOKKOS_ARCH})
  set(_pgas_default_kokkos_arch "$ENV{PGAS_KOKKOS_ARCH}")
endif()
set(PGAS_KOKKOS_ARCH "${_pgas_default_kokkos_arch}" CACHE STRING "Kokkos NVIDIA GPU architecture suffix, e.g. TURING75, AMPERE80.")

set(_pgas_default_cuda_architectures "75")
if(DEFINED ENV{PGAS_CUDA_ARCHITECTURES})
  set(_pgas_default_cuda_architectures "$ENV{PGAS_CUDA_ARCHITECTURES}")
endif()
set(PGAS_CUDA_ARCHITECTURES "${_pgas_default_cuda_architectures}" CACHE STRING "CMake CUDA architectures, e.g. 75, 80, 86, 89, 90.")

if(Kokkos_ENABLE_CUDA)
  if(PGAS_KOKKOS_ARCH)
    set(Kokkos_ARCH_${PGAS_KOKKOS_ARCH} ON CACHE BOOL "PGAS-selected Kokkos CUDA architecture" FORCE)
  endif()
  if(PGAS_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${PGAS_CUDA_ARCHITECTURES}" CACHE STRING "PGAS-selected CUDA architectures" FORCE)
  endif()
endif()

FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 4.3.01
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(Kokkos)
