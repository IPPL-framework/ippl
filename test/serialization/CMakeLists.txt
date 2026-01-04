# -----------------------------------------------------------------------------
# test/serialization/CMakeLists.txt
#
# Integration test for IPPL serialization functionality.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding serialization integration tests from: ${_relPath}")

# === Individual tests ===
add_ippl_integration_test(serialize01 LABELS serialization integration)
