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
#   [COMPILE_ONLY]                     # compile and link but do not register/run test
#   [ARGS <arg1> <arg2> ...]           # args passed to the test binary
#   [MPI_ARGS <arg1> <arg2> ...]       # extra args for mpiexec
#   [NUM_PROCS <N>]                    # default: IPPL_DEFAULT_TEST_PROCS (2)
#   [TIMEOUT <sec>]                    # default: 60
#   [WORKING_DIRECTORY <dir>]          # default: current binary dir
#   [LABELS <lbl1> <lbl2> ...]         # default: unit
#   [LAUNCH <tool> [tool-args...]]     # e.g. LAUNCH "valgrind;--leak-check=full"
#   [NO_MPI]                           # run without mpiexec
#   [REQUIRE_MPI]                      # disable test if MPI not found
#   [RUN_SERIAL]                       # ctest runs this test serially
#   [USE_GTEST_MAIN]                   # link GTest::gtest_main instead of gtest
#   [LINK_LIBS <lib1> <lib2> ...]      # extra link libs
#   [INCLUDE_DIRS <dir1> <dir2> ...]   # extra include dirs for this target
#   [PROPERTIES <ctest-prop> <val> ...]# extra set_tests_properties
#   [INTEGRATION]                      # mark test as integration
# )
# ~~~
# -----------------------------------------------------------------------------

set(IPPL_DEFAULT_TEST_PROCS "2" CACHE STRING "Default MPI ranks per unit test")
set(IPPL_DEFAULT_TEST_TIMEOUT "60" CACHE STRING "Default timeout (seconds) per unit test")

function(add_ippl_test TEST_NAME)
  set(options NO_MPI REQUIRE_MPI RUN_SERIAL USE_GTEST_MAIN INTEGRATION COMPILE_ONLY)
  set(oneValueArgs NUM_PROCS TIMEOUT WORKING_DIRECTORY)
  set(multiValueArgs
      LABELS
      ARGS
      MPI_ARGS
      SOURCES
      LAUNCH
      LINK_LIBS
      INCLUDE_DIRS
      PROPERTIES)
  cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)
  set(CTEST_TEST_NAME "${TEST_NAME}")
  if("${TEST_NAME}" IN_LIST IPPL_DISABLED_TEST_LIST_${_build_type})
    message(STATUS "Marking disabled test: ${TEST_NAME}")
    set(CTEST_TEST_NAME "known_fail_${TEST_NAME}")
  endif()

  if(TEST_SOURCES)
    set(_sources ${TEST_SOURCES})
  else()
    set(_sources ${TEST_NAME}.cpp)
  endif()

  # ensure the test itself is compiled and linked
  add_executable(${TEST_NAME} ${_sources})

  # Link libraries (ippl exports includes/flags/deps; pick your GTest flavor)
  if(TEST_USE_GTEST_MAIN)
    target_link_libraries(${TEST_NAME} PRIVATE IPPL::ippl GTest::gtest_main)
  else()
    target_link_libraries(${TEST_NAME} PRIVATE IPPL::ippl GTest::gtest)
  endif()

  if(TEST_INTEGRATION)
    target_link_libraries(${TEST_NAME} PRIVATE IPPL::ippl ${TEST_LINK_LIBS})
  endif()

  if(TARGET ippl_build_flags)
    target_link_libraries(${TEST_NAME} PRIVATE ippl_build_flags)
  endif()

  if(TARGET ippl::test_support)
    target_link_libraries(${TEST_NAME} PRIVATE ippl::test_support)
  endif()

  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_INCLUDE_DIRS})

  # MPI ranks
  if(TEST_NUM_PROCS)
    set(_procs "${TEST_NUM_PROCS}")
  else()
    set(_procs "${IPPL_DEFAULT_TEST_PROCS}")
  endif()

  if(TEST_TIMEOUT)
    set(_timeout "${TEST_TIMEOUT}")
  else()
    set(_timeout "${IPPL_DEFAULT_TEST_TIMEOUT}")
  endif()

  if(TEST_INTEGRATION)
    set(_labels integration)
  else()
    set(_labels unit)
  endif()

  if(TEST_LABELS)
    list(APPEND _labels ${TEST_LABELS})
  endif()

  # Build the test command
  set(_cmd $<TARGET_FILE:${TEST_NAME}>)
  if(TEST_ARGS)
    list(APPEND _cmd ${TEST_ARGS})
  endif()

  # Optional launcher (e.g., valgrind)
  if(TEST_LAUNCH)
    set(_launched_cmd ${TEST_LAUNCH} ${_cmd})
  else()
    set(_launched_cmd ${_cmd})
  endif()

  # Parallel-ctest friendliness: processors = ranks * threads
  set(_threads "${IPPL_OPENMP_THREADS}")
  if(NOT _threads)
    set(_threads 1)
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
      add_test(NAME ${CTEST_TEST_NAME} COMMAND ${_launched_cmd})
      set_tests_properties(${CTEST_TEST_NAME} PROPERTIES DISABLED TRUE SKIP_REGULAR_EXPRESSION
                                                         "MPI required but NOT found")
      return()
    else()
      # Fallback: run single-process without mpiexec
      message(
        STATUS
          "add_ippl_test (${CTEST_TEST_NAME}): MPI NOT found; running TEST ${TEST_NAME} without mpiexec"
      )
      set(_final_cmd ${_launched_cmd})
    endif()
  endif()

  # Name prefix for nicer grouping: unit.<relpath>.<name> file(RELATIVE_PATH _rel
  # "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}") string(REPLACE "/" "." _rel "${_rel}")
  set(_ctest_name "${CTEST_TEST_NAME}")

  # Register the test
  if(BUILD_TESTING AND NOT TEST_COMPILE_ONLY)
    add_test(NAME ${_ctest_name} COMMAND ${_final_cmd})

    # Base properties
    set_tests_properties(${_ctest_name} PROPERTIES TIMEOUT ${_timeout} LABELS "${_labels}")

    if(TEST_INTEGRATION)
      math(EXPR _processors "${_procs}*${_threads}")
      # TODO: We might not want to set FFTW/MKL/BLAS threads to all available
      set(ENV_VARS)
      list(
        APPEND
        ENV_VARS
        "OMP_PROC_BIND=spread"
        "OMP_PLACES=threads"
        "OMP_NUM_THREADS=${_threads} "
        "KOKKOS_NUM_THREADS=${_threads}"
        "MKL_NUM_THREADS=${_threads}"
        "OPENBLAS_NUM_THREADS=${_threads}"
        "FFTW_THREADS=${_threads}")

      # Set processors, working directory, and environment for parallel tests
      set_tests_properties(${_ctest_name} PROPERTIES PROCESSORS ${_processors} ENVIRONMENT
                                                     "${ENV_VARS}")
    endif()

    # Optional working directory
    if(TEST_WORKING_DIRECTORY)
      set(_workdir "${TEST_WORKING_DIRECTORY}")
    else()
      if(TEST_INTEGRATION)
        set(_workdir "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_work")
      else()
        set(_workdir "${CMAKE_CURRENT_BINARY_DIR}")
      endif()
    endif()
    file(MAKE_DIRECTORY "${_workdir}")
    set_tests_properties(${_ctest_name} PROPERTIES WORKING_DIRECTORY "${_workdir}")

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
