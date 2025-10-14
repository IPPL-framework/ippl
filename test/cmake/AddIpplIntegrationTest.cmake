# -----------------------------------------------------------------------------
# AddIpplIntegrationTest.cmake
#
# Defines a helper function `add_ippl_integration_test()` to add integration tests. It builds an
# executable, links it with IPPL and MPI, includes headers from IPPL, and registers the test with
# CTest. Labels are optional and default to "integration".
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ~~~
# add_ippl_integration_test(<name>
#   [SOURCES <src1> <src2> ...]        # default: <name>.cpp
#   [ARGS <arg1> <arg2> ...]           # args passed to the test binary
#   [MPI_ARGS <arg1> <arg2> ...]       # extra args for mpiexec
#   [NUM_PROCS <N>]                    # default: IPPL_DEFAULT_TEST_PROCS (2)
#   [TIMEOUT <sec>]                    # default: 300
#   [WORKING_DIRECTORY <dir>]          # default: per-test build dir
#   [LABELS <lbl1> <lbl2> ...]         # default: integration
#   [LAUNCH <tool> [tool-args...]]     # e.g. LAUNCH "valgrind;--leak-check=full"
#   [NO_MPI]                           # run without mpiexec
#   [REQUIRE_MPI]                      # disable test if MPI not found
#   [LINK_LIBS <lib1> <lib2> ...]      # extra link libs
#   [INCLUDE_DIRS <dir1> <dir2> ...]   # extra include dirs for this target
#   [PROPERTIES <ctest-prop> <val> ...]# extra set_tests_properties
# )
# ~~~
# -----------------------------------------------------------------------------

set(IPPL_DEFAULT_TEST_PROCS "2" CACHE STRING "Default MPI ranks per unit/integration test")

function(add_ippl_integration_test TEST_NAME)
  set(options NO_MPI REQUIRE_MPI)
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

  target_link_libraries(${TEST_NAME} PRIVATE ippl::ippl ${TEST_LINK_LIBS})
  if(TARGET ippl_build_flags)
    target_link_libraries(${TEST_NAME} PRIVATE ippl_build_flags)
  endif()

  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_INCLUDE_DIRS})

  # Ranks
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

  set(_labels integration)
  if(TEST_LABELS)
    list(APPEND _labels ${TEST_LABELS})
  endif()

  # Build the command: binary [ARGS...]
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

  # MPI handling
  if(TEST_NO_MPI)
    set(_final_cmd ${_launched_cmd})
  else()
    if(DEFINED MPIEXEC_EXECUTABLE)
      set(_final_cmd ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_procs} ${MPIEXEC_PREFLAGS}
                     ${TEST_MPI_ARGS} ${_launched_cmd})
    elseif(TEST_REQUIRE_MPI)
      # Create a disabled test with a clear reason
      set(_final_cmd ${_launched_cmd})
      set(_will_disable TRUE)
    else()
      message(
        STATUS "add_ippl_integration_test(${TEST_NAME}): MPI not found; running without mpiexec")
      set(_final_cmd ${_launched_cmd})
    endif()
  endif()

  # Parallel-ctest friendliness: processors = ranks * threads
  set(_threads "$ENV{OMP_NUM_THREADS}")
  if(NOT _threads)
    set(_threads 1)
  endif()
  math(EXPR _processors "${_procs} * ${_threads}")

  # Unique working directory
  if(TEST_WORKING_DIRECTORY)
    set(_workdir "${TEST_WORKING_DIRECTORY}")
  else()
    set(_workdir "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_work")
  endif()
  file(MAKE_DIRECTORY "${_workdir}")

  set(_ctest_name "${TEST_NAME}")

  if(BUILD_TESTING)
    add_test(NAME ${_ctest_name} COMMAND ${_final_cmd})

    # Core properties
    set_tests_properties(
      ${_ctest_name}
      PROPERTIES
        TIMEOUT
        ${_timeout}
        LABELS
        "${_labels}"
        PROCESSORS
        ${_processors}
        WORKING_DIRECTORY
        "${_workdir}"
        ENVIRONMENT
        "OMP_NUM_THREADS=${_threads};
         KOKKOS_NUM_THREADS=${_threads};
         MKL_NUM_THREADS=1;
         OPENBLAS_NUM_THREADS=1;
         FFTW_THREADS=1")

    # Disable if MPI was required but missing
    if(_will_disable)
      set_tests_properties(${_ctest_name} PROPERTIES DISABLED TRUE SKIP_REGULAR_EXPRESSION
                                                     "MPI required but not found")
    endif()

    # Extra CTest properties from caller
    if(TEST_PROPERTIES)
      set_tests_properties(${_ctest_name} PROPERTIES ${TEST_PROPERTIES})
    endif()
  endif()
endfunction()
