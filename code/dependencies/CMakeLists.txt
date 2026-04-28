include(FetchContent)

FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 4.3.01
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(Kokkos)