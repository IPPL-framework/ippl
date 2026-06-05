# -----------------------------------------------------------------------------
# test/solver/multigrid/CMakeLists.txt
#
# Integration tests for IPPL CG solver preconditioned with multigrid.
# -----------------------------------------------------------------------------


file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding multigrid solver integration tests from: ${_relPath}")


# tests the CG solver with Multigrid Preconditioner (Size 16^3, Test Case 1: Periodic)
add_ippl_integration_test(TestMultigrid ARGS 4 1 LABELS solver integration)


add_ippl_integration_test(TestMultigrid_debug COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_convergence COMPILE_ONLY LABELS solver integration)
