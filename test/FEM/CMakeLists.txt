# -----------------------------------------------------------------------------
# test/FEM/CMakeLists.txt
#
# Integration tests for FEM functionality. These build example executables
# that run and validate FEM behavior across supported configurations.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding FEM integration tests from: ${_relPath}")

add_ippl_integration_test(TestLagrangeSpace1D LABELS integration fem)
add_ippl_integration_test(TestLagrangeSpace2D LABELS integration fem)
