set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)
if(PORT STREQUAL "openblas")
    # Baseline ARMv8; do not require the build runner's extensions (such as SVE).
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS -DTARGET=ARMV8)
endif()
