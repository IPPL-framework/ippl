# IPPL has already resolved the complete host link line. Preserve absolute paths
# and static transitive dependencies; upstream 5.2's finder treats them as names.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TPLLAPACKE REQUIRED_VARS LAPACKE_INCLUDE_DIRS LAPACKE_LIBRARIES)
set(_lapackeResolved)
foreach(_library IN LISTS LAPACKE_LIBRARIES)
  if(TARGET "${_library}")
    list(APPEND _lapackeResolved "${_library}")
  elseif(_library MATCHES "^-")
    # CMake LAPACK providers may return linker items such as -lm or -pthread.
    # Preserve them verbatim instead of treating them as find_library names.
    list(APPEND _lapackeResolved "${_library}")
  elseif(IS_ABSOLUTE "${_library}")
    if(NOT EXISTS "${_library}")
      message(FATAL_ERROR "LAPACKE dependency does not exist: ${_library}")
    endif()
    list(APPEND _lapackeResolved "${_library}")
  else()
    unset(_lapackeLibrary CACHE)
    find_library(_lapackeLibrary NAMES ${_library} HINTS ${LAPACKE_LIBRARY_DIRS} REQUIRED)
    list(APPEND _lapackeResolved "${_lapackeLibrary}")
  endif()
endforeach()
set(_lapackeIncludes "${LAPACKE_INCLUDE_DIRS}")
if(IPPL_HOST_LAPACK_FETCHED)
  # The imported host target carries relocatable include directories itself.
  set(_lapackeIncludes)
endif()
kokkoskernels_create_imported_tpl(LAPACKE INTERFACE
  LINK_LIBRARIES "${_lapackeResolved}")
if(_lapackeIncludes)
  # Environment views may also contain an incompatible desul installation.
  # Keep their broad include directory behind Kokkos's bundled TPL headers.
  target_include_directories(LAPACKE SYSTEM INTERFACE ${_lapackeIncludes})
endif()

if(IPPL_HOST_LAPACK_FETCHED)
  # Also make the Kernels package usable directly, including in a fresh IPPL
  # build that finds this Kernels installation instead of fetching its sources.
  kokkoskernels_append_config_line("include(CMakeFindDependencyMacro)")
  kokkoskernels_append_config_line(
    "find_dependency(IPPLHostLapack CONFIG HINTS \"\\\${PACKAGE_PREFIX_DIR}\" \"${IPPL_HOST_LAPACK_PREFIX}\")")
endif()
