# -----------------------------------------------------------------------------
# src/LinearSolvers/CMakeLists.txt
#
# Adds linear solver interface headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install LinearSolvers headers
install(FILES
    SolverAlgorithm.h
    Preconditioner.h
    PCG.h
    DESTINATION include/LinearSolvers
)
