# -----------------------------------------------------------------------------
# test/solver/CMakeLists.txt
#
# Integration tests for IPPL solvers.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding solver integration tests from: ${_relPath}")

# tests the CG solver
add_ippl_integration_test(TestCGSolver ARGS 4 j LABELS solver integration)

# compile only
add_ippl_integration_test(TestSolverDesign COMPILE_ONLY LABELS solver integration)
add_ippl_integration_test(TestCGSolver_convergence COMPILE_ONLY LABELS solver integration)

if(IPPL_ENABLE_FFT)
  # tests FFTPeriodicPoissonSolver
  add_ippl_integration_test(TestFFTPeriodicPoissonSolver LABELS solver integration)

  # tests FFTOpenPoissonSolver
  add_ippl_integration_test(TestGaussian ARGS 16 16 16 pencils a2a no-reorder HOCKNEY LABELS solver integration)

  # tests the FFTTruncatedGreenPeriodicPoissonSolver
  add_ippl_integration_test(TestFFTTruncatedGreenPeriodicPoissonSolver ARGS 16 16 16 LABELS solver integration)

  # compile only
  add_ippl_integration_test(TestGaussian_convergence COMPILE_ONLY LABELS solver integration)
  add_ippl_integration_test(TestSphere COMPILE_ONLY LABELS solver integration)
  add_ippl_integration_test(Budiardja_plot COMPILE_ONLY LABELS solver integration)
  add_ippl_integration_test(TestGaussian_biharmonic COMPILE_ONLY LABELS solver integration TIMEOUT 120)
  add_ippl_integration_test(TestGaussian_hessian COMPILE_ONLY LABELS solver integration)
endif()
