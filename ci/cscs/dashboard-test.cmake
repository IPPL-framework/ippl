cmake_minimum_required(VERSION 3.20)

set(DASHBOARD_PROJECT "IPPL")

if(NOT DEFINED BUILD_TYPE)
  set(BUILD_TYPE Debug)
endif()

if(NOT DEFINED BUILD_DIR)
  message(FATAL_ERROR "BUILD_DIR must be defined")
endif()

if(NOT DEFINED CTEST_BUILD_NAME)
  message(FATAL_ERROR "CTEST_BUILD_NAME must be defined")
endif()

# --- CDash metadata must match build ---
set(CTEST_SITE "${CTEST_SITE}")
set(CTEST_BUILD_CONFIGURATION ${BUILD_TYPE})
set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}")

set(CTEST_SOURCE_DIRECTORY "$ENV{CI_PROJECT_DIR}")
set(CTEST_BINARY_DIRECTORY "${BUILD_DIR}")
set(CTEST_CMAKE_GENERATOR "Ninja")
set(CTEST_GROUP "Pull_Requests")
set(CTEST_GROUP "Experimental")

# --- append to the existing dashboard entry ---
ctest_start(Experimental GROUP "${CTEST_GROUP}" APPEND)

# --- redirect .gcda files into the test job's build directory ---
# The executables were built in a different CI job with a different absolute path. gcov writes .gcda
# files next to the absolute .gcno path by default, so without this redirection CTest coverage would
# find no .gcda files.
if(ENABLE_COVERAGE)
  set(ENV{GCOV_PREFIX} "${CTEST_BINARY_DIRECTORY}")
  string(REPLACE "/" ";" _gcovPrefixList "${CTEST_BINARY_DIRECTORY}")
  list(LENGTH _gcovPrefixList _gcovPrefixLen)
  math(EXPR _gcovPrefixStrip "${_gcovPrefixLen} - 1")
  set(ENV{GCOV_PREFIX_STRIP} "${_gcovPrefixStrip}")
  message(
    STATUS
      "Redirecting .gcda files to ${CTEST_BINARY_DIRECTORY} (GCOV_PREFIX_STRIP=${_gcovPrefixStrip})"
  )
endif()

# --- run tests : we use srun and already control parallelism
ctest_test(PARALLEL_LEVEL 1 RETURN_VALUE test_result)

# --- collect coverage if requested ---
if(ENABLE_COVERAGE)
  message(STATUS "Collecting coverage with gcov")
  ctest_read_custom_files("${CTEST_SOURCE_DIRECTORY}")
  set(CTEST_COVERAGE_COMMAND "gcov")
  ctest_coverage(RETURN_VALUE cov_result)
endif()

# --- submit test (and coverage) results ---
ctest_submit()

# --- fail if any test failed ---
if(test_result)
  message(FATAL_ERROR "CTest reported test failures")
endif()

string(ASCII 27 ESC)
set(BLUE "${ESC}[34m")
set(RESET "${ESC}[0m")
message("${BLUE}# ---------------------------------${RESET}")
message("${BLUE}To view ALL configure/build/test results and error logs visit: ${RESET}")
message("${BLUE}https://my.cdash.org/index.php?project=${DASHBOARD_PROJECT}${RESET}")
message("${BLUE}For this PR visit: ${RESET}")
message(
  "${BLUE}https://my.cdash.org/index.php?project=${DASHBOARD_PROJECT}&filtercount=1&showfilters=1&field1=buildname&compare1=63&value1=${CDASH_LABEL}${RESET}"
)
message("${BLUE}# ---------------------------------${RESET}")
