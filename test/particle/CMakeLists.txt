# -----------------------------------------------------------------------------
# test/particle/CMakeLists.txt
#
# Integration tests and benchmarks related to the Particle system in IPPL.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding Particle integration tests from: ${_relPath}")

add_ippl_integration_test(TestScatter             LABELS integration particle)
add_ippl_integration_test(TestGather              LABELS integration particle)
add_ippl_integration_test(PICnd                   LABELS integration particle)
add_ippl_integration_test(benchmarkParticleUpdate LABELS integration particle)
