# -----------------------------------------------------------------------------
# validation check for LandauDamping test
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
  # Build the C++ correctness validation test
  add_executable(LandauDampingCorrectness LandauDampingCorrectness.cpp)

  message("Adding test: LandauDampingCorrectness")

  # command line params are : output, reference, tolerance
  add_test(
    NAME LandauDampingCorrectnessValidation
    COMMAND LandauDampingCorrectness "${PROJECT_BINARY_DIR}/alpine/data/FieldLandau_2_manager.csv"
            "${PROJECT_SOURCE_DIR}/alpine/validation/FieldLandau_valid_result.csv" "4E-1"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/alpine/")

  set_tests_properties(
    LandauDampingCorrectnessValidation
    PROPERTIES LABELS "validation;alpine" TIMEOUT 30
               # This test should only run if the LandauDamping integration test passed
               DEPENDS LandauDamping)
endif()
