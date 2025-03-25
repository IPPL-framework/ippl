# -----------------------------------------------------------------------------
# src/Interpolation/CMakeLists.txt
#
# Adds particle-to-grid interpolation headers to the IPPL target.
# -----------------------------------------------------------------------------

# Make headers accessible via <Interpolation/...> if needed
target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install Interpolation-related headers
install(FILES
    CIC.h
    CIC.hpp
    DESTINATION include/Interpolation
)

