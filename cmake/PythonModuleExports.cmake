# Keep private numerical symbols out of the Python modules' public exports.
# Linux uses an ELF version script; Windows uses explicit PyInit exports.
function(csp_isolate_python_module target)
 get_target_property(kind "${target}" TYPE)
 if(NOT kind STREQUAL "MODULE_LIBRARY")
  message(FATAL_ERROR "Unsupported Python-module isolation target/platform")
 endif()
 if(WIN32)
  # PE imports bind to specific DLLs. Only pybind11's explicit PyInit export is
  # public; do not enable CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS.
  set_target_properties(${target} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  return()
 elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Python-module isolation is implemented for Linux and Windows")
 endif()
 if(NOT target MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR "Invalid Python extension entry-point name")
 endif()
 set(exports "${CMAKE_CURRENT_BINARY_DIR}/${target}.exports")
 file(GENERATE OUTPUT "${exports}" CONTENT "{ global: PyInit_${target}; local: *; };\n")
 target_link_options("${target}" PRIVATE "LINKER:--version-script,${exports}")
 set_property(TARGET "${target}" APPEND PROPERTY LINK_DEPENDS "${exports}")
endfunction()
