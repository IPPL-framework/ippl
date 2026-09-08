# IPPL has already resolved the complete host link line. Preserve absolute paths
# and static transitive dependencies; upstream 5.2's finder treats them as names.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TPLLAPACKE REQUIRED_VARS LAPACKE_INCLUDE_DIRS LAPACKE_LIBRARIES)
set(_lapackeResolved)
foreach(_library IN LISTS LAPACKE_LIBRARIES)
  if(IS_ABSOLUTE "${_library}")
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
kokkoskernels_create_imported_tpl(LAPACKE INTERFACE
  LINK_LIBRARIES "${_lapackeResolved}" INCLUDES "${LAPACKE_INCLUDE_DIRS}")
