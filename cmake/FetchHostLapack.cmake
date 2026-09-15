# Bootstrap the host library at configure time so the normal link/runtime probes
# still validate a real library on the first configure. Subsequent builds reuse it.
function(ippl_fetch_host_lapack)
  if(CMAKE_CROSSCOMPILING AND NOT IPPL_LAPACKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "LAPACKE fallback in a cross build needs IPPL_LAPACKE_TOOLCHAIN_FILE for the target CPU, or an installed LAPACKE provider.")
  endif()
  set(IPPL_LAPACKE_BUILD_JOBS 4 CACHE STRING "Parallel jobs for reference LAPACK bootstrap")
  set(IPPL_LAPACKE_TOOLCHAIN_FILE "" CACHE FILEPATH "Optional host LAPACK C/Fortran toolchain")
  FetchContent_Declare(ippl_reference_lapack
    URL https://codeload.github.com/Reference-LAPACK/lapack/tar.gz/refs/tags/v3.12.1
    URL_HASH SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR ippl-bootstrap-only)
  FetchContent_MakeAvailable(ippl_reference_lapack)
  set(_build "${FETCHCONTENT_BASE_DIR}/ippl-host-lapack-${IPPL_LAPACK_INTEGER_BYTES}-build")
  set(_prefix "${FETCHCONTENT_BASE_DIR}/ippl-host-lapack-${IPPL_LAPACK_INTEGER_BYTES}-install")
  set(_options)
  # Let the host project discover C independently (or use CC/the toolchain).
  # IPPL may enable C only later through a dependency, so forwarding its C
  # compiler would change the bootstrap compiler on the second configure.
  foreach(_variable CMAKE_Fortran_COMPILER CMAKE_OSX_ARCHITECTURES CMAKE_OSX_SYSROOT CMAKE_OSX_DEPLOYMENT_TARGET)
    if(DEFINED ${_variable} AND NOT "${${_variable}}" STREQUAL "")
      if(_variable MATCHES "_COMPILER$")
        list(APPEND _options "-D${_variable}:FILEPATH=${${_variable}}")
      else()
        list(APPEND _options "-D${_variable}:STRING=${${_variable}}")
      endif()
    endif()
  endforeach()
  if(IPPL_LAPACKE_TOOLCHAIN_FILE)
    list(APPEND _options "-DCMAKE_TOOLCHAIN_FILE=${IPPL_LAPACKE_TOOLCHAIN_FILE}")
  endif()
  message(STATUS "Host LAPACKE not found: building reference LAPACK 3.12.1 (C/Fortran); logs in ${_build}")
  file(MAKE_DIRECTORY "${_build}")
  execute_process(COMMAND "${CMAKE_COMMAND}"
    -S "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/lapack-bootstrap" -B "${_build}"
    "-DIPPL_REFERENCE_LAPACK_SOURCE_DIR=${ippl_reference_lapack_SOURCE_DIR}"
    "-DIPPL_LAPACK_INTEGER_BYTES=${IPPL_LAPACK_INTEGER_BYTES}"
    "-DCMAKE_INSTALL_PREFIX=${_prefix}" -DCMAKE_BUILD_TYPE=Release ${_options}
    RESULT_VARIABLE _result OUTPUT_FILE "${_build}/configure.log" ERROR_FILE "${_build}/configure.log")
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Host LAPACKE bootstrap configure failed. A host C and Fortran compiler (e.g. gfortran) is required. See ${_build}/configure.log")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" --build "${_build}" --config Release
    --parallel "${IPPL_LAPACKE_BUILD_JOBS}"
    RESULT_VARIABLE _result OUTPUT_FILE "${_build}/build.log" ERROR_FILE "${_build}/build.log")
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Host LAPACKE bootstrap build failed; see ${_build}/build.log")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" --install "${_build}" --config Release
    RESULT_VARIABLE _result OUTPUT_FILE "${_build}/install.log" ERROR_FILE "${_build}/install.log")
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Host LAPACKE bootstrap install failed; see ${_build}/install.log")
  endif()
  # Use this exact bootstrap, including after a change of integer ABI.
  set(IPPLHostLapack_DIR "${_prefix}/lib/cmake/IPPLHostLapack")
  find_package(IPPLHostLapack CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${IPPLHostLapack_DIR}")
  # Imported targets created inside a function retain their directory visibility.
  # Bundle only the private archives, public headers and our relocatable package;
  # upstream pkg-config files contain the bootstrap prefix and are not exported.
  install(DIRECTORY "${_prefix}/include/ippl-lapack" DESTINATION include)
  install(DIRECTORY "${_prefix}/lib/ippl-lapack/" DESTINATION lib/ippl-lapack
          FILES_MATCHING PATTERN "*.a" PATTERN "*.lib"
          PATTERN "cmake" EXCLUDE PATTERN "pkgconfig" EXCLUDE)
  install(FILES "${_prefix}/lib/cmake/IPPLHostLapack/IPPLHostLapackConfig.cmake"
          DESTINATION lib/cmake/IPPLHostLapack)
  install(FILES "${_prefix}/share/ippl/LAPACK-LICENSE" DESTINATION share/ippl)
  set(LAPACKE_INCLUDE_DIRS "${_prefix}/include/ippl-lapack" PARENT_SCOPE)
  set(LAPACKE_LIBRARIES IPPLHostLapack::lapacke PARENT_SCOPE)
  set(IPPL_HOST_LAPACK_PREFIX "${_prefix}" PARENT_SCOPE)
  set(IPPL_HOST_LAPACK_FETCHED ON PARENT_SCOPE)
endfunction()
