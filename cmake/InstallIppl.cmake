# -----------------------------------------------------------------------------
# InstallIppl.cmake
# ~~~
#
# Setup install rules for ippl (and dependencies if needed).
#
# Responsibilities:
#   - Create IPPLConfig, IPPLConfigVersion, IPPLTargets files in install dir
#   - Also create above files in build dir, for development, when building ippl
#     locally and using a non-installed version
#
# Note:
#   - NameSpace 'IPPL' should be used to match Project name as used in
#     find_package(IPPL ...
# ~~~
# -----------------------------------------------------------------------------

set(IPPL_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/ippl"
    CACHE PATH "Directory for ippl CMake package files")

# cmake-format: off
# -------------------------------------------------------
# Define patterns for globbing of files we are not installing
# -------------------------------------------------------
set(_ippl_install_excludes
    PATTERN "CMakeFiles"     EXCLUDE
    PATTERN "CMakeLists.txt" EXCLUDE
    PATTERN "*.c"   EXCLUDE
    PATTERN "*.cc"  EXCLUDE
    PATTERN "*.cpp" EXCLUDE
    PATTERN "*.cu"  EXCLUDE
    # match your filenames if needed
)

if(NOT IPPL_ENABLE_FFT)
  list(
    APPEND _ippl_install_excludes
    PATTERN "FFT/*" EXCLUDE
    PATTERN "PoissonSolvers/FFT*" EXCLUDE)
endif()

# -------------------------------------------------------
# Define patterns for files we are installing
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

# -------------------------------------------------------
# install all the header/other files
# -------------------------------------------------------
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/IpplVersions.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# -------------------------------------------------------
# Define an install rule for the targets we export
# -------------------------------------------------------
install(
  TARGETS ippl
  EXPORT ipplTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # static libs, import libs
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # shared libs
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # executables (if any)
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# -------------------------------------------------------
# Install rule to create the CMake targets for find_package()
# -------------------------------------------------------
install(
  EXPORT ipplTargets
  FILE IPPLTargets.cmake
  NAMESPACE IPPL::
  DESTINATION ${IPPL_INSTALL_CMAKEDIR})

# Also export targets to the build directory to support uninstalled builds
export(
  EXPORT ipplTargets
  FILE "${PROJECT_BINARY_DIR}/IPPLTargets.cmake"
  NAMESPACE IPPL::)

# cmake-format: on

# -------------------------------------------------------
# Create the project ConfigVersion file
# -------------------------------------------------------
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/IPPLConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

# Also generate version file in the build directory to support uninstalled builds
write_basic_package_version_file("${PROJECT_BINARY_DIR}/IPPLConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

# -------------------------------------------------------
# Setup the main project config file found by find_package(IPPL
# -------------------------------------------------------
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/IPPLConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/IPPLConfig.cmake"
  INSTALL_DESTINATION ${IPPL_INSTALL_CMAKEDIR})

# Also generate config file in the build directory to support uninstalled builds
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/IPPLConfig.cmake.in" "${PROJECT_BINARY_DIR}/IPPLConfig.cmake"
  INSTALL_DESTINATION lib/cmake/IPPL)

# -------------------------------------------------------
# The install rule that copies the generated config files to the install tree
# -------------------------------------------------------
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/IPPLConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/IPPLConfigVersion.cmake"
        DESTINATION ${IPPL_INSTALL_CMAKEDIR})

# -------------------------------------------------------
# Fix/Hack: Ensure extern dependencies are exported correctly if they were built in-tree. This is
# needed for Heffte because it doesn't fully use CMake's export target mechanism
# -------------------------------------------------------
if(TARGET Heffte)
  install(TARGETS Heffte EXPORT ipplTargets DESTINATION lib)
endif()
