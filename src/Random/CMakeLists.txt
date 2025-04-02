# -----------------------------------------------------------------------------
# src/Random/CMakeLists.txt
#
# Adds random sampling and distribution headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install Random-related headers
install(FILES
    InverseTransformSampling.h
    Utility.h
    Distribution.h
    UniformDistribution.h
    NormalDistribution.h
    Randn.h
    Randu.h
    DESTINATION include/Random
)

