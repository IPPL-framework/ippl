# -----------------------------------------------------------------------------
# src/MaxwellSolvers/CMakeLists.txt
#
# Adds Maxwell solver header to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install MaxwellSolvers header
install(FILES
    Maxwell.h
    FDTDSolverBase.h
    FDTDSolverBase.hpp
    StandardFDTDSolver.h
    StandardFDTDSolver.hpp
    NonStandardFDTDSolver.h
    NonStandardFDTDSolver.hpp
    AbsorbingBC.h
    DESTINATION include/MaxwellSolvers
)

