# -----------------------------------------------------------------------------
# test/solver/CMakeLists.txt
#
# Integration tests for IPPL solvers.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding solver integration tests from: ${_relPath}")

add_ippl_integration_test(TestSolverDesign LABELS solver integration)
add_ippl_integration_test(TestCGSolver LABELS solver integration)

if(IPPL_ENABLE_FFT)
    add_ippl_integration_test(TestGaussian_convergence LABELS solver integration)
    add_ippl_integration_test(TestSphere LABELS solver integration)
    add_ippl_integration_test(Budiardja_plot LABELS solver integration)
    add_ippl_integration_test(TestGaussian LABELS solver integration)
    add_ippl_integration_test(TestFFTPeriodicPoissonSolver LABELS solver integration)
    add_ippl_integration_test(TestGaussian_biharmonic LABELS solver integration)
    add_ippl_integration_test(TestGaussian_hessian LABELS solver integration)
    add_ippl_integration_test(TestFFTTruncatedGreenPeriodicPoissonSolver LABELS solver integration)
endif()
