# cmake/InstallIppl.cmake
# -------------------------------------------------------
# Installation logic for the IPPL library
# -------------------------------------------------------

set(IPPL_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ippl"
    CACHE PATH "Directory for ippl CMake package files")

# Install public headers
install(FILES
    ${IPPL_SOURCE_DIR}/Ippl.h
    ${IPPL_SOURCE_DIR}/IpplCore.h
    ${IPPL_BINARY_DIR}/IpplVersions.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the actual library target
install(TARGETS ippl
    EXPORT IpplTargets
    ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}     # static libs, import libs
    LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}     # shared libs
    RUNTIME     DESTINATION ${CMAKE_INSTALL_BINDIR}     # executables (if any)
    INCLUDES    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export the CMake target for find_package()
install(EXPORT IpplTargets
    FILE IpplTargets.cmake
    NAMESPACE Ippl::
    DESTINATION ${IPPL_INSTALL_CMAKEDIR}
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ipplConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/IpplConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/IpplConfig.cmake"
    INSTALL_DESTINATION ${IPPL_INSTALL_CMAKEDIR}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/IpplConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/IpplConfigVersion.cmake"
    DESTINATION ${IPPL_INSTALL_CMAKEDIR}
)

