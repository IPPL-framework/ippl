# -----------------------------------------------------------------------------
# src/PoissonSolvers/CMakeLists.txt
#
# Adds Poisson solver headers to the IPPL target, optionally with FFT backends.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Default Poisson solver headers
set(poisson_headers
    PoissonCG.h
    Poisson.h
    NullSolver.h
    LaplaceHelpers.h
    FEMPoissonSolver.h
    PreconditionedFEMPoissonSolver.h
)

# Add FFT-based solver headers conditionally
if(IPPL_ENABLE_FFT)
    list(APPEND poisson_headers
        FFTOpenPoissonSolver.h
        FFTOpenPoissonSolver.hpp
        FFTPeriodicPoissonSolver.h
        FFTPeriodicPoissonSolver.hpp
        FFTTruncatedGreenPeriodicPoissonSolver.h
        FFTTruncatedGreenPeriodicPoissonSolver.hpp
    )
endif()

# Install Poisson solver headers
install(FILES ${poisson_headers} DESTINATION include/PoissonSolvers)

