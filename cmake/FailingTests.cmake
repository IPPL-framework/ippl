# -----------------------------------------------------------------------------
# FailingTests.cmake
#
# Maintains a list of tests that are excluded from testing due to the fact that they are known to
# fail and spoil an otherwise green dashboard.
#
# Fixing these tests should be considered a priority
#
# -----------------------------------------------------------------------------

if(BUILD_TESTING AND IPPL_SKIP_FAILING_TESTS)
  set(IPPL_DISABLED_TEST_LIST
      ParticleSendRecv
      ORB
      PIC
      TestSolverDesign
      TestGaussian_convergence
      TestSphere
      Budiardja_plot
      TestGaussian
      TestGaussian_hessian
      TestFFTTruncatedGreenPeriodicPoissonSolver
      TestScaling_ZeroBC_sin
      TestScaling_PeriodicBC_sinsin
      TestMaxwellDiffusionZeroBC
      TestMaxwellDiffusionPolyZeroBC
      TestMaxwellDiffusionPolyZeroBCTimed
      TestScaling_ZeroBC_sin_precon
      TestScaling_PeriodicBC_sinsin_precon
      TestNonStandardFDTDSolver_convergence)
endif()
