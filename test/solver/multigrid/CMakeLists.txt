# -----------------------------------------------------------------------------
# test/solver/multigrid/CMakeLists.txt
#
# Integration tests for IPPL CG solver preconditioned with multigrid.
# -----------------------------------------------------------------------------


file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding multigrid solver integration tests from: ${_relPath}")


# tests the CG solver with Multigrid Preconditioner (Size 16^3, Test Case 1: Periodic)
add_ippl_integration_test(TestMultigrid ARGS 4 1 LABELS solver integration)

# Convergence test for CG preconditioned with the multigrid method
add_ippl_integration_test(TestMultigrid_discrete_constant COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_discrete_periodic COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_convergence_constant COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_convergence_constant_anisotropic COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_convergence_periodic COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_convergence_periodic_anisotropic COMPILE_ONLY LABELS solver integration)

# Tolerance studies for multigrid method

add_ippl_integration_test(TestMultigrid_tolerance_constant COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_tolerance_constant_anisotropic COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_tolerance_periodic COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_tolerance_periodic_anisotropic COMPILE_ONLY LABELS solver integration)

# Comparison Multigrid to other Preconditioners
add_ippl_integration_test(TestMultigrid_comparison_constant COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_comparison_periodic COMPILE_ONLY LABELS solver integration)

add_ippl_integration_test(TestMultigrid_debug COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestMultigrid_convergence COMPILE_ONLY LABELS solver integration)


