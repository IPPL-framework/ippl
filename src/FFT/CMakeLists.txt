# -----------------------------------------------------------------------------
# src/FFT/CMakeLists.txt
#
# Adds FFT interface headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install FFT-related headers
install(FILES
    FFT.hpp
    FFT.h
    DESTINATION include/FFT
)

