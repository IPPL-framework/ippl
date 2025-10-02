# -----------------------------------------------------------------------------
# test/types/CMakeLists.txt
#
# Integration tests for IPPL types (Vector, Tuple, etc.).
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding type-related integration tests from: ${_relPath}")

add_ippl_integration_test(TestVector LABELS integration types)
# add_ippl_integration_test(TestTuple LABELS integration types)
add_ippl_integration_test(TestVector2 LABELS integration types)
add_ippl_integration_test(TestVector3 LABELS integration types)
