# Opt-in Linux native-core profile; the ordinary installation rules stay intact.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" OR _pgas_build_gpu)
    message(FATAL_ERROR "Native-core installation is qualified only for Linux x86_64 CPU")
endif()
set(CSP_NATIVE_CORE_NOTICES_DIR "" CACHE PATH "Admitted third-party notices and hash manifest")
if(NOT EXISTS "${CSP_NATIVE_CORE_NOTICES_DIR}/MANIFEST.json")
    message(FATAL_ERROR "Native-core installation requires an admitted notice manifest")
endif()

# Preserve build-tree lookup, but never inherit build/provider paths at install.
set_target_properties(pgas_bound_cpu PROPERTIES
    INSTALL_RPATH "" INSTALL_RPATH_USE_LINK_PATH FALSE BUILD_WITH_INSTALL_RPATH FALSE)
install(TARGETS pgas_bound_cpu
    LIBRARY DESTINATION c_spikes/pgas COMPONENT CSPNativeCore
    RUNTIME DESTINATION c_spikes/pgas COMPONENT CSPNativeCore)
if(TARGET oasis_methods)
    set_target_properties(oasis_methods PROPERTIES
        INSTALL_RPATH "" INSTALL_RPATH_USE_LINK_PATH FALSE BUILD_WITH_INSTALL_RPATH FALSE)
    install(TARGETS oasis_methods
        LIBRARY DESTINATION c_spikes/oasis COMPONENT CSPNativeCore
        RUNTIME DESTINATION c_spikes/oasis COMPONENT CSPNativeCore)
endif()
install(FILES "${CMAKE_SOURCE_DIR}/src/c_spikes/__init__.py"
    DESTINATION c_spikes COMPONENT CSPNativeCore)
install(FILES "${CMAKE_SOURCE_DIR}/src/c_spikes/pgas/__init__.py"
    DESTINATION c_spikes/pgas COMPONENT CSPNativeCore)
install(FILES "${CMAKE_SOURCE_DIR}/src/c_spikes/oasis/__init__.py"
    "${CMAKE_SOURCE_DIR}/src/c_spikes/oasis/functions.py"
    DESTINATION c_spikes/oasis COMPONENT CSPNativeCore)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
    DESTINATION share/c_spikes COMPONENT CSPNativeCore)
install(FILES "${CMAKE_SOURCE_DIR}/src/c_spikes/oasis/PROVENANCE.md"
    DESTINATION share/c_spikes RENAME OASIS_PROVENANCE.md COMPONENT CSPNativeCore)
install(DIRECTORY "${CSP_NATIVE_CORE_NOTICES_DIR}/"
    DESTINATION share/c_spikes/notices COMPONENT CSPNativeCore)
