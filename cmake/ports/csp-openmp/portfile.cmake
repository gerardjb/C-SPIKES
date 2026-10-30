vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)
vcpkg_download_distfile(OPENMP_ARCHIVE
    URLS "https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/openmp-19.1.7.src.tar.xz"
    FILENAME "openmp-19.1.7.src.tar.xz"
    SHA512 a6ee7e672d6f175bab672fd7cc03914485f528fc6ff82a6d8e10192de6480263bbc99dbff373d68f68a16edf0a7b46982d0fc4198c2069ff807bb77464fece7a)
vcpkg_download_distfile(CMAKE_ARCHIVE
    URLS "https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/cmake-19.1.7.src.tar.xz"
    FILENAME "cmake-19.1.7.src.tar.xz"
    SHA512 3ff56e1a61dbace35d8c168ad5c94c5a09d92eb683494bfe1deabcb3640cf79b7422d44903d049d2298d54c2874f9a60c4f13f6795a3949e7eaefd5d42e62621)
vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${OPENMP_ARCHIVE}" SOURCE_BASE openmp)
vcpkg_extract_source_archive(CMAKE_SOURCE_PATH ARCHIVE "${CMAKE_ARCHIVE}" SOURCE_BASE llvm-cmake)
# Standalone OpenMP expects LLVM's common CMake modules in a sibling directory.
file(COPY "${CMAKE_SOURCE_PATH}/" DESTINATION "${SOURCE_PATH}/../cmake")
vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}" OPTIONS
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF
    -DLIBOMP_OMPT_SUPPORT=OFF
    -DLIBOMP_ENABLE_SHARED=ON
    -DOPENMP_MSVC_NAME_SCHEME=OFF)
vcpkg_cmake_install()
vcpkg_copy_pdbs()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.TXT")
