include(FetchContent)

FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 8e15454876298832f013ebcaf08d53825b5bda40 # 5.2.2
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(Kokkos)
