# -----------------------------------------------------------------------------
# src/FFT/CMakeLists.txt
#
# Adds FFT interface headers to the IPPL target.
# -----------------------------------------------------------------------------

target_link_libraries(ippl PRIVATE Heffte::Heffte)

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install FFT-related headers
install(FILES
    FFT.hpp
    FFT.h
    DESTINATION include/FFT
)

