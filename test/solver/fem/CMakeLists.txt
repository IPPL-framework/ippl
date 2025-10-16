# -----------------------------------------------------------------------------
# test/solver/fem/CMakeLists.txt
#
# Integration tests for IPPL FEM solvers.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding FEM solver integration tests from: ${_relPath}")

add_ippl_integration_test(TestZeroBC_sin LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_sin_preconditioned LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_constant1d LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_constant1d_preconditioned LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_polynomial2d LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_polynomial2d_preconditioned LABELS solver fem integration)
add_ippl_integration_test(TestPeriodicBC_sin LABELS solver fem integration)
add_ippl_integration_test(TestPeriodicBC_sin_preconditioned LABELS solver fem integration)
add_ippl_integration_test(TestPeriodicBC_sinsin LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_1d LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_2d LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_3d LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_1d_preconditioned LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_2d_preconditioned LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_3d_preconditioned LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_1dcos LABELS solver fem integration)
add_ippl_integration_test(TestScaling_ZeroBC_sin LABELS solver fem integration)
add_ippl_integration_test(TestScaling_PeriodicBC_sinsin LABELS solver fem integration)
add_ippl_integration_test(TestMaxwellDiffusionZeroBC LABELS solver fem integration)
add_ippl_integration_test(TestMaxwellDiffusionPolyZeroBC LABELS solver fem integration)
add_ippl_integration_test(TestMaxwellDiffusionPolyZeroBCTimed LABELS solver fem integration)
add_ippl_integration_test(TestScaling_ZeroBC_sin_precon LABELS solver fem integration)
add_ippl_integration_test(TestScaling_PeriodicBC_sinsin_precon LABELS solver fem integration)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables: mode: cmake cmake-tab-width: 4 indent-tabs-mode: nil require-final-newline: nil
# End:
