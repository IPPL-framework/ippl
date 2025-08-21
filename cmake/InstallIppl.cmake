# cmake/InstallIppl.cmake
# -------------------------------------------------------
# Installation logic for the IPPL library
# -------------------------------------------------------

set(IPPL_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ippl"
    CACHE PATH "Directory for ippl CMake package files")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl
        FILES_MATCHING
          PATTERN "*.h"   PATTERN "*.hpp"   PATTERN "*.hh"   PATTERN "*.H"
          PATTERN "*.cuh" PATTERN "*.tpp"
        # Exclude build/system files and sources
          PATTERN "CMakeFiles" EXCLUDE
          PATTERN "CMakeLists.txt" EXCLUDE
          PATTERN "*.c" EXCLUDE
          PATTERN "*.cc" EXCLUDE
          PATTERN "*.cpp" EXCLUDE
          PATTERN "*.cu" EXCLUDE)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/IpplVersions.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the actual library target
install(TARGETS ippl
    EXPORT ipplTargets
    ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}     # static libs, import libs
    LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}     # shared libs
    RUNTIME     DESTINATION ${CMAKE_INSTALL_BINDIR}     # executables (if any)
    INCLUDES    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export the CMake target for find_package()
install(EXPORT ipplTargets
    FILE ipplTargets.cmake
    NAMESPACE ippl::
    DESTINATION ${IPPL_INSTALL_CMAKEDIR}
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ipplConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/ipplConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ipplConfig.cmake"
    INSTALL_DESTINATION ${IPPL_INSTALL_CMAKEDIR}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ipplConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ipplConfigVersion.cmake"
    DESTINATION ${IPPL_INSTALL_CMAKEDIR}
)

