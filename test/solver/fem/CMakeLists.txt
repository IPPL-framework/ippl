# -----------------------------------------------------------------------------
# test/solver/fem/CMakeLists.txt
#
# Integration tests for IPPL FEM solvers.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding FEM solver integration tests from: ${_relPath}")

add_ippl_integration_test(TestZeroBC_sin LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_constant1d LABELS solver fem integration)
add_ippl_integration_test(TestZeroBC_polynomial2d LABELS solver fem integration)
add_ippl_integration_test(TestPeriodicBC_sin LABELS solver fem integration)
add_ippl_integration_test(TestPeriodicBC_sinsin LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_1d LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_2d LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_3d LABELS solver fem integration)
add_ippl_integration_test(TestNonhomDirichlet_1dcos LABELS solver fem integration)
add_ippl_integration_test(TestScaling_ZeroBC_sin LABELS solver fem integration)
add_ippl_integration_test(TestScaling_PeriodicBC_sinsin LABELS solver fem integration)


add_executable(TestFEMSolverZeroBC_sin TestFEMSolverZeroBC_sin.cpp)
target_link_libraries(
    TestFEMSolverZeroBC_sin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestFEMVectorZeroBC_sin TestFEMVectorZeroBC_sin.cpp)
target_link_libraries(
    TestFEMVectorZeroBC_sin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestMaxwellDiffusionZeroBC TestMaxwellDiffusionZeroBC.cpp)
target_link_libraries(
    TestMaxwellDiffusionZeroBC
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestMaxwellDiffusionPolyZeroBC TestMaxwellDiffusionPolyZeroBC.cpp)
target_link_libraries(
    TestMaxwellDiffusionPolyZeroBC
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestMaxwellDiffusionPolyZeroBCTimed TestMaxwellDiffusionPolyZeroBCTimed.cpp)
target_link_libraries(
    TestMaxwellDiffusionPolyZeroBCTimed
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)


# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
