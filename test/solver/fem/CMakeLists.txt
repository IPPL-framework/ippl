# -----------------------------------------------------------------------------
# test/solver/fem/CMakeLists.txt
#
# Integration tests for IPPL FEM solvers.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding FEM solver integration tests from: ${_relPath}")

# cmake-format: off

# ----------------------------------------------------------------------------
# one dimensional Zero BC tests
add_ippl_integration_test(TestZeroBC_sin 
    ARGS 1 LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_sin_preconditioned 
    ARGS 1 LABELS solver fem integration)

# one dimensional Periodic BC tests
add_ippl_integration_test(TestPeriodicBC_sin 
    ARGS 1 LABELS solver fem integration TIMEOUT 60)
add_ippl_integration_test(TestPeriodicBC_sin_preconditioned 
    ARGS 1 LABELS solver fem integration TIMEOUT 60)

# one dimensional Non-homogeneous Dirichlet BC tests
add_ippl_integration_test(TestNonhomDirichlet_1d 
    LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_1d_preconditioned 
    LABELS solver fem integration)

# ----------------------------------------------------------------------------
# Compile only (no testing)

add_ippl_integration_test(TestPeriodicBC_sinsin 
    COMPILE_ONLY LABELS solver fem integration)

add_ippl_integration_test(TestZeroBC_constant1d 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_constant1d_preconditioned 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_polynomial2d 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_polynomial2d_preconditioned 
    COMPILE_ONLY LABELS solver fem integration)

add_ippl_integration_test(TestNonhomDirichlet_2d 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_3d 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_2d_preconditioned 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_3d_preconditioned 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_1dcos 
    COMPILE_ONLY LABELS solver fem integration)

add_ippl_integration_test(TestScaling_ZeroBC_sin 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestScaling_PeriodicBC_sinsin 
    COMPILE_ONLY LABELS solver fem integration)

add_ippl_integration_test(TestMaxwellDiffusionPolyZeroBC 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestMaxwellDiffusionZeroBC 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestMaxwellDiffusionPolyZeroBCTimed 
    COMPILE_ONLY LABELS solver fem integration)

add_ippl_integration_test(TestScaling_ZeroBC_sin_precon 
    COMPILE_ONLY LABELS solver fem integration)
add_ippl_integration_test(TestScaling_PeriodicBC_sinsin_precon 
    COMPILE_ONLY LABELS solver fem integration)
# cmake-format: on
