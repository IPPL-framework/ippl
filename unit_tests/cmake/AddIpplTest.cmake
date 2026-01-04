# -----------------------------------------------------------------------------
# AddIpplTest.cmake
#
# Defines a helper macro `add_ippl_test()` to create a unit test executable. It links to IPPL and
# GoogleTest, and sets include directories and labels.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ~~~
# add_ippl_test(<name>
#   [SOURCES <src1> <src2> ...]        # default: <name>.cpp
#   [ARGS <arg1> <arg2> ...]           # args passed to the test binary
#   [MPI_ARGS <arg1> <arg2> ...]       # extra args for mpiexec
#   [NUM_PROCS <N>]                    # default: IPPL_DEFAULT_TEST_PROCS (2)
#   [TIMEOUT <sec>]                    # default: 300
#   [WORKING_DIRECTORY <dir>]          # default: current binary dir
#   [LABELS <lbl1> <lbl2> ...]         # default: unit
#   [LAUNCH <tool> [tool-args...]]     # e.g. LAUNCH "valgrind;--leak-check=full"
#   [NO_MPI]                           # run without mpiexec
#   [REQUIRE_MPI]                      # disable test if MPI not found
#   [RUN_SERIAL]                       # ctest runs this test serially
#   [USE_GTEST_MAIN]                   # link GTest::gtest_main instead of gtest
#   [PROPERTIES <ctest-prop> <val> ...]# extra set_tests_properties
# )
# ~~~
# -----------------------------------------------------------------------------

set(IPPL_DEFAULT_TEST_PROCS "2" CACHE STRING "Default MPI ranks per unit test")

function(add_ippl_test TEST_NAME)
  set(options NO_MPI REQUIRE_MPI RUN_SERIAL USE_GTEST_MAIN)
  set(oneValueArgs NUM_PROCS TIMEOUT WORKING_DIRECTORY)
  set(multiValueArgs LABELS ARGS MPI_ARGS SOURCES LAUNCH PROPERTIES)
  cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if("${TEST_NAME}" IN_LIST IPPL_DISABLED_TEST_LIST)
    message(STATUS "Skipping disabled test: ${TEST_NAME}")
    return()
  endif()

  if(TEST_SOURCES)
    set(_sources ${TEST_SOURCES})
  else()
    set(_sources ${TEST_NAME}.cpp)
  endif()

  add_executable(${TEST_NAME} ${_sources})

  # Link libraries (ippl exports includes/flags/deps; pick your GTest flavor)
  if(TEST_USE_GTEST_MAIN)
    target_link_libraries(${TEST_NAME} PRIVATE IPPL::ippl GTest::gtest_main)
  else()
    target_link_libraries(${TEST_NAME} PRIVATE IPPL::ippl GTest::gtest)
  endif()

  if(TARGET ippl::test_support)
    target_link_libraries(${TEST_NAME} PRIVATE ippl::test_support)
  endif()

  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # MPI ranks
  if(TEST_NUM_PROCS)
    set(_procs "${TEST_NUM_PROCS}")
  else()
    set(_procs "${IPPL_DEFAULT_TEST_PROCS}")
  endif()

  if(TEST_TIMEOUT)
    set(_timeout "${TEST_TIMEOUT}")
  else()
    set(_timeout 300)
  endif()

  set(_labels unit)
  if(TEST_LABELS)
    list(APPEND _labels ${TEST_LABELS})
  endif()

  # Build the test command
  set(_cmd $<TARGET_FILE:${TEST_NAME}>)
  if(TEST_ARGS)
    list(APPEND _cmd ${TEST_ARGS})
  endif()

  if(TEST_LAUNCH)
    set(_launched_cmd ${TEST_LAUNCH} ${_cmd})
  else()
    set(_launched_cmd ${_cmd})
  endif()

  # MPI handling
  set(_final_cmd)
  if(TEST_NO_MPI)
    set(_final_cmd ${_launched_cmd})
  else()
    if(DEFINED MPIEXEC_EXECUTABLE)
      set(_final_cmd ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_procs} ${MPIEXEC_PREFLAGS}
                     ${TEST_MPI_ARGS} ${_launched_cmd})
    elseif(TEST_REQUIRE_MPI)
      # Add a disabled test with a clear message
      add_test(NAME ${TEST_NAME} COMMAND ${_launched_cmd})
      set_tests_properties(${TEST_NAME} PROPERTIES DISABLED TRUE SKIP_REGULAR_EXPRESSION
                                                   "MPI required but not found")
      return()
    else()
      # Fallback: run single-process without mpiexec
      message(STATUS "add_ippl_test(${TEST_NAME}): MPI not found; running without mpiexec")
      set(_final_cmd ${_launched_cmd})
    endif()
  endif()

  # Name prefix for nicer grouping: unit.<relpath>.<name> file(RELATIVE_PATH _rel
  # "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}") string(REPLACE "/" "." _rel "${_rel}")
  set(_ctest_name "${TEST_NAME}")

  # Register the test
  if(BUILD_TESTING)
    add_test(NAME ${_ctest_name} COMMAND ${_final_cmd})

    # Base properties
    set_tests_properties(${_ctest_name} PROPERTIES TIMEOUT ${_timeout} LABELS "${_labels}")

    # Optional working directory
    if(TEST_WORKING_DIRECTORY)
      set_tests_properties(${_ctest_name} PROPERTIES WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}")
    else()
      set_tests_properties(${_ctest_name} PROPERTIES WORKING_DIRECTORY
                                                     "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    # Run serially if requested
    if(TEST_RUN_SERIAL)
      set_tests_properties(${_ctest_name} PROPERTIES RUN_SERIAL TRUE)
    endif()

    # Extra user-specified CTest properties (if any)
    if(TEST_PROPERTIES)
      set_tests_properties(${_ctest_name} PROPERTIES ${TEST_PROPERTIES})
    endif()
  endif()
endfunction()
