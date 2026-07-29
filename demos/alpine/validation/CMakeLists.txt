# -----------------------------------------------------------------------------
# validation check for LandauDamping test
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
  # Build the C++ correctness validation test
  add_executable(LandauDampingCorrectness LandauDampingCorrectness.cpp)

  message("Adding test: LandauDampingCorrectness")

  # Keep the reference data in the build tree so test artifacts remain relocatable between CI jobs.
  set(LANDAU_DAMPING_REFERENCE_CSV
      "${CMAKE_CURRENT_BINARY_DIR}/FieldLandau_valid_result.csv")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/FieldLandau_valid_result.csv"
                 "${LANDAU_DAMPING_REFERENCE_CSV}" COPYONLY)

  # command line params are : output, reference, tolerance
  add_test(
    NAME LandauDampingCorrectnessValidation
    COMMAND LandauDampingCorrectness "${PROJECT_BINARY_DIR}/alpine/data/FieldLandau_2_manager.csv"
            "${LANDAU_DAMPING_REFERENCE_CSV}" "4E-1"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/alpine/")

  set_tests_properties(
    LandauDampingCorrectnessValidation
    PROPERTIES LABELS "validation;alpine" TIMEOUT 30
               # This test should only run if the LandauDamping integration test passed
               DEPENDS LandauDamping)
endif()
