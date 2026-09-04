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

# --- run tests : we use srun and already control parallelism
ctest_test(PARALLEL_LEVEL 1 RETURN_VALUE test_result)

# --- collect coverage if requested ---
set(_coverage_symlink "")
if(ENABLE_COVERAGE)
  message(STATUS "Collecting coverage with gcov")
  ctest_read_custom_files("${CTEST_SOURCE_DIRECTORY}")

  # gcov records the absolute source path from the build job.  If the current CI_PROJECT_DIR differs
  # from that path, create a temporary symlink so that gcov can find the source files during
  # coverage processing.  The symlink is removed again as soon as coverage collection finishes.
  set(_cache_file "${BUILD_DIR}/CMakeCache.txt")
  if(EXISTS "${_cache_file}")
    file(STRINGS "${_cache_file}" _cache_home_line REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
    if(_cache_home_line)
      list(GET _cache_home_line 0 _cache_home_line)
      string(REGEX REPLACE "^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$" "\\1" _build_source_dir
                           "${_cache_home_line}")
      if(NOT _build_source_dir STREQUAL CTEST_SOURCE_DIRECTORY)
        message(
          STATUS
            "Creating temporary source symlink: ${_build_source_dir} -> ${CTEST_SOURCE_DIRECTORY}")
        get_filename_component(_build_source_parent "${_build_source_dir}" DIRECTORY)
        file(MAKE_DIRECTORY "${_build_source_parent}")
        file(REMOVE_RECURSE "${_build_source_dir}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CTEST_SOURCE_DIRECTORY}"
                                "${_build_source_dir}")
        set(_coverage_symlink "${_build_source_dir}")
      endif()
    endif()
  endif()

  set(CTEST_COVERAGE_COMMAND "gcov")
  ctest_coverage(RETURN_VALUE cov_result)

  if(_coverage_symlink)
    file(REMOVE "${_coverage_symlink}")
  endif()
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
