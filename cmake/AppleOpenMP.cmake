# AppleClang understands OpenMP but Xcode does not ship the runtime or headers.
# Use the same pinned LLVM runtime source as our Windows builds.
set(_csp_openmp_root "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
find_library(_csp_openmp_library NAMES omp
    PATHS "${_csp_openmp_root}/lib" NO_DEFAULT_PATH REQUIRED)
foreach(language C CXX)
    set(OpenMP_${language}_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "AppleClang OpenMP flags" FORCE)
    set(OpenMP_${language}_LIB_NAMES "omp" CACHE STRING "Pinned LLVM OpenMP runtime" FORCE)
    set(OpenMP_${language}_INCLUDE_DIR "${_csp_openmp_root}/include" CACHE PATH "LLVM OpenMP headers" FORCE)
endforeach()
set(OpenMP_omp_LIBRARY "${_csp_openmp_library}" CACHE FILEPATH "Pinned LLVM OpenMP runtime" FORCE)
