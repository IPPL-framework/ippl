# -----------------------------------------------------------------------------
# test/random/CMakeLists.txt
#
# Integration tests for the random sampling infrastructure in IPPL.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding random integration tests from: ${_relPath}")

add_ippl_integration_test(TestInverseTransformSamplingNormal LABELS random integration)

add_ippl_integration_test(TestInverseTransformSamplingUpdateBounds LABELS random integration)

add_ippl_integration_test(TestInverseTransformSamplingSpecificRange LABELS random integration)

add_ippl_integration_test(TestInverseTransformSamplingCustom LABELS random integration)

add_ippl_integration_test(TestUnboundedNormal LABELS random integration)
