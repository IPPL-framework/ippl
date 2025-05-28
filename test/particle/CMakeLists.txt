# -----------------------------------------------------------------------------
# test/particle/CMakeLists.txt
#
# Integration tests and benchmarks related to the Particle system in IPPL.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding Particle integration tests from: ${_relPath}")

add_ippl_integration_test(TestScatter             LABELS integration particle)
add_ippl_integration_test(TestGather              LABELS integration particle)

add_ippl_integration_test(PICnd                   
    LABELS integration particle
    ARGS 128 128 128 10000 10 --info 10
)

add_ippl_integration_test(benchmarkParticleUpdate 
  LABELS integration particle
  ARGS 128 128 128 10000 10 --info 10
)


#add_ippl_integration_test(univ-3
#  LABELS integration particle
#  ARGS 128 128 128 10000 10 --info 10
#)

#add_ippl_integration_test(univ-2
#  LABELS integration particle
#  ARGS 128 128 128 10000 10 --info 10
#)

add_ippl_integration_test(zeldo-test1
  LABELS integration particle
  ARGS 128 128 128 10000 10 --info 10
)

add_ippl_integration_test(zeldo-test-mp1
  LABELS integration particle
  ARGS 128 128 128 10000 10 --info 10
)
