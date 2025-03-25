# cmake/InstallIppl.cmake
# -------------------------------------------------------
# Installation logic for the IPPL library
# -------------------------------------------------------

# Install public headers
install(FILES
    ${IPPL_SOURCE_DIR}/Ippl.h
    ${IPPL_SOURCE_DIR}/IpplCore.h
    ${IPPL_BINARY_DIR}/IpplVersions.h
    DESTINATION include
)

# Install the actual library target
install(TARGETS ippl
    EXPORT IpplTargets
    DESTINATION lib
)

# Export the CMake target for find_package()
install(EXPORT IpplTargets
    FILE IpplTargets.cmake
    NAMESPACE Ippl::
    DESTINATION lib/cmake/Ippl
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/IpplConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/IpplConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/IpplConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Ippl
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/IpplConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/IpplConfigVersion.cmake"
    DESTINATION lib/cmake/Ippl
)

