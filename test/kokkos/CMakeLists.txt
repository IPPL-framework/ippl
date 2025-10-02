# -----------------------------------------------------------------------------
# test/field/CMakeLists.txt
#
# Integration tests for kokkos-related functionality.
# -----------------------------------------------------------------------------
file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding field integration tests from: ${_relPath}")

add_ippl_integration_test(TestVectorField LABELS integration kokkos)
add_ippl_integration_test(TestVectorField2 LABELS integration kokkos)
add_ippl_integration_test(TestVectorField3 LABELS integration kokkos)
add_ippl_integration_test(TestVectorField4 LABELS integration kokkos)
