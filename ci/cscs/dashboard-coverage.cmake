cmake_minimum_required(VERSION 3.20)

set(DASHBOARD_PROJECT "IPPL")

if(NOT DEFINED BUILD_TYPE)
  set(BUILD_TYPE Debug)
endif()

if(NOT DEFINED CDASH_LABEL)
  set(CDASH_LABEL "branch")
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

set(_ci_project_dir "$ENV{CI_PROJECT_DIR}")
set(CTEST_SOURCE_DIRECTORY "${_ci_project_dir}")
set(CTEST_BINARY_DIRECTORY "${BUILD_DIR}")
set(CTEST_CMAKE_GENERATOR "Ninja")
set(CTEST_GROUP "Experimental")

# --- gcov must match the compiler that produced the .gcda/.gcno files ---
# The uenv holding the build compiler is only mounted inside `srun --uenv` steps, so this script has
# to run inside such a step (see ci/cscs/openmp/build_openmp.yml).  A gcov taken from the job shell
# is typically a different gcc version; it cannot read the profiling data and then silently reports
# 0% coverage.
if(GCOV_COMMAND)
  if(NOT EXISTS "${GCOV_COMMAND}")
    message(
      FATAL_ERROR "GCOV_COMMAND '${GCOV_COMMAND}' does not exist, run inside the uenv srun step")
  endif()
  set(CTEST_COVERAGE_COMMAND "${GCOV_COMMAND}")
else()
  set(CTEST_COVERAGE_COMMAND "gcov")
endif()

# --- append to the existing dashboard entry ---
ctest_start(Experimental GROUP "${CTEST_GROUP}" APPEND)

message(STATUS "Collecting coverage with gcov")
ctest_read_custom_files("${CTEST_SOURCE_DIRECTORY}")

file(GLOB_RECURSE _gcno_files "${CTEST_BINARY_DIRECTORY}/*.gcno")
file(GLOB_RECURSE _gcda_files "${CTEST_BINARY_DIRECTORY}/*.gcda")
list(LENGTH _gcno_files _gcno_count)
list(LENGTH _gcda_files _gcda_count)
message(STATUS "Coverage input: ${_gcno_count} .gcno files, ${_gcda_count} .gcda files")

execute_process(COMMAND ${CTEST_COVERAGE_COMMAND} --version OUTPUT_VARIABLE _gcov_version
                ERROR_QUIET)
string(REGEX REPLACE "\n.*" "" _gcov_version "${_gcov_version}")
message(STATUS "Using gcov command: ${CTEST_COVERAGE_COMMAND} (${_gcov_version})")

ctest_coverage(RETURN_VALUE cov_result)
if(cov_result)
  message(WARNING "ctest_coverage returned ${cov_result}")
endif()

# --- submit coverage results ---
ctest_submit(RETURN_VALUE submit_result)
if(submit_result)
  message(
    WARNING "ctest_submit failed, coverage results remain under ${CTEST_BINARY_DIRECTORY}/Testing")
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
