# -----------------------------------------------------------------------------
# src/Decomposition/CMakeLists.txt
#
# Adds headers for domain decomposition strategies to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install public headers
install(FILES
    OrthogonalRecursiveBisection.h
    OrthogonalRecursiveBisection.hpp
    DESTINATION include/Decomposition
)

