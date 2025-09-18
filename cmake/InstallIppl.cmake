# cmake/InstallIppl.cmake
# -------------------------------------------------------
# Installation logic for the IPPL library
# -------------------------------------------------------

set(IPPL_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/ippl"
    CACHE PATH "Directory for ippl CMake package files")

set(_ippl_install_excludes
    PATTERN
    "CMakeFiles"
    EXCLUDE
    PATTERN
    "CMakeLists.txt"
    EXCLUDE
    PATTERN
    "*.c"
    EXCLUDE
    PATTERN
    "*.cc"
    EXCLUDE
    PATTERN
    "*.cpp"
    EXCLUDE
    PATTERN
    "*.cu"
    EXCLUDE)
if(NOT IPPL_ENABLE_FFT)
  list(
    APPEND
    _ippl_install_excludes
    PATTERN
    "FFT/*"
    EXCLUDE
    PATTERN
    "PoissonSolvers/FFT*"
    EXCLUDE # match your filenames if needed
  )
endif()

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.hh"
  PATTERN "*.H"
  PATTERN "*.cuh"
  PATTERN "*.tpp" ${_ippl_install_excludes})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/IpplVersions.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the actual library target
install(
  TARGETS ippl
  EXPORT ipplTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # static libs, import libs
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # shared libs
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # executables (if any)
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the CMake target for find_package()
install(EXPORT ipplTargets FILE ipplTargets.cmake NAMESPACE ippl::
        DESTINATION ${IPPL_INSTALL_CMAKEDIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/IPPLConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/IPPLConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/IPPLConfig.cmake"
  INSTALL_DESTINATION ${IPPL_INSTALL_CMAKEDIR})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/IPPLConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/IPPLConfigVersion.cmake"
        DESTINATION ${IPPL_INSTALL_CMAKEDIR})

return()

# cmake/InstallIppl.cmake
# -------------------------------------------------------
# Installation logic for the IPPL library
# -------------------------------------------------------
include(CMakePackageConfigHelpers)

# -------------------------------------------------------
# Install public headers
install(FILES ${IPPL_SOURCE_DIR}/Ippl.h ${IPPL_SOURCE_DIR}/IpplCore.h
              ${IPPL_BINARY_DIR}/IpplVersions.h DESTINATION include)

# -------------------------------------------------------
# Install the actual library target
install(TARGETS ippl EXPORT IpplTargets DESTINATION lib)

# Ensure extern dependencies (like HEFFTE) are exported correctly if they were built in-tree. This
# is necessary for Heffte because it does not use CMake's export target mechanism
if(TARGET Heffte)
  install(TARGETS Heffte EXPORT IpplTargets DESTINATION lib)
endif()

# -------------------------------------------------------
# Export the CMake targets for find_package()
install(EXPORT IpplTargets FILE IPPLTargets.cmake NAMESPACE IPPL:: DESTINATION lib/cmake/IPPL)

# -------------------------------------------------------
# Also export targets to the build directory to support uninstalled builds
export(EXPORT IpplTargets FILE "${PROJECT_BINARY_DIR}/IPPLTargets.cmake" NAMESPACE IPPL::)

# -------------------------------------------------------
# generate a version file for the package (in the build tree)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/IPPLConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

# -------------------------------------------------------
# generate a package config file (in the build tree)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/IPPLConfig.cmake.in" "${PROJECT_BINARY_DIR}/IPPLConfig.cmake"
  INSTALL_DESTINATION lib/cmake/IPPL)

# -------------------------------------------------------
# install the generated version and package config files in the install tree
install(FILES "${PROJECT_BINARY_DIR}/IPPLConfig.cmake"
              "${PROJECT_BINARY_DIR}/IPPLConfigVersion.cmake" DESTINATION lib/cmake/IPPL)
