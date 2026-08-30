# -----------------------------------------------------------------------------
# demos/alpine/CMakeLists.txt
#
# This module provides example physics applications using IPPL.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/alpine/")

add_subdirectory(ExamplesWithoutPicManager)

function(add_alpine_example name)
  add_executable(${name} ${name}.cpp)

  target_link_libraries(${name} PRIVATE IPPL::ippl)

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

if(IPPL_ENABLE_TESTS)
  # cmake-format: off

  set(LANDAU_RANKS 2)
  set(LANDAU_CSV "${PROJECT_BINARY_DIR}/demos/alpine/data/FieldLandau_${LANDAU_RANKS}_manager.csv")

  # Create the data directory and remove any stale CSV so the LandauDamping
  # fixture starts clean. We use bash instead of ${CMAKE_COMMAND} because the
  # build and test phases may run with different uenv mount points, making the
  # absolute cmake path from the build phase unavailable at test time.
  add_test(NAME LandauDampingInit
    COMMAND bash -c "mkdir -p '${PROJECT_BINARY_DIR}/demos/alpine/data' && rm -f '${LANDAU_CSV}'")
  set_tests_properties(LandauDampingInit PROPERTIES FIXTURES_SETUP landau_fixture)

  # Add the test
  add_ippl_integration_test(LandauDamping 
    ARGS "16" "16" "16" "10000000" "25" "FFT" "0.01" "LeapFrog" "--overallocate" "2.0" "--info" "10"
    LABELS alpine integration
    NUM_PROCS ${LANDAU_RANKS}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/demos/alpine/")

  set_tests_properties(LandauDamping PROPERTIES FIXTURES_REQUIRED landau_fixture)
    
  # cmake-format: on
else()
  add_alpine_example(LandauDamping)
endif()

add_alpine_example(PenningTrap)
add_alpine_example(BumponTailInstability)

add_subdirectory(validation)

if(IPPL_ENABLE_BENCHMARK)
  include(AddIpplBenchmark)

  add_ippl_benchmark(
    LandauDampingBench
    SOURCES
    LandauDampingBench.cpp
    REPORT_TIMERS
    solve
    pushVelocity
    pushPosition
    update
    loadBalance
    ARGS
    "16"
    "16"
    "16"
    "100000"
    "5"
    "FFT"
    "0.01"
    "LeapFrog"
    "--overallocate"
    "2.0"
    "--info"
    "10"
    NUM_PROCS
    2
    LABELS
    alpine)
endif()
