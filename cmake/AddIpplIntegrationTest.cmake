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
#   [COMPILE_ONLY]                     # compile and link but do not register/run test
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
include(AddIpplTest)

function(add_ippl_integration_test TEST_NAME)
  add_ippl_test(${TEST_NAME} INTEGRATION ${ARGN})
endfunction()
