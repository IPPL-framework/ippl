# -----------------------------------------------------------------------------
# test/region/CMakeLists.txt
#
# Integration tests for IPPL region classes (NDRegion, PRegion, etc.).
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding region integration tests from: ${_relPath}")

add_ippl_integration_test(TestPRegion LABELS region integration)

add_ippl_integration_test(TestNDRegion LABELS region integration)
