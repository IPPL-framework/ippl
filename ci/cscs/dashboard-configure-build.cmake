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

# The full build name here gets overwritten in test phase, so leave this blank for now
set(TEST_INFO "")

# --- CDash metadata ---
set(CTEST_SITE "${CTEST_SITE}")
set(CTEST_BUILD_CONFIGURATION ${BUILD_TYPE})
set(CTEST_BUILD_NAME "${CDASH_LABEL}-${BUILD_ARCH}-${BUILD_TYPE}-${TEST_INFO}")

set(CTEST_SOURCE_DIRECTORY "$ENV{CI_PROJECT_DIR}")
set(CTEST_BINARY_DIRECTORY "${BUILD_DIR}")
set(CTEST_CMAKE_GENERATOR "Ninja")
set(CTEST_GROUP "Experimental")

# --- start a new build in CDash ---
ctest_start(Experimental GROUP "${CTEST_GROUP}")

# --- Initialize base configure command as a CMake LIST ---
set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND}")
string(APPEND CTEST_CONFIGURE_COMMAND " -S${CTEST_SOURCE_DIRECTORY}")
string(APPEND CTEST_CONFIGURE_COMMAND " -B${CTEST_BINARY_DIRECTORY}")
string(APPEND CTEST_CONFIGURE_COMMAND " -G${CTEST_CMAKE_GENERATOR}")
string(APPEND CTEST_CONFIGURE_COMMAND " --preset=${PRESET}")
string(APPEND CTEST_CONFIGURE_COMMAND " -DCMAKE_BUILD_TYPE=${BUILD_TYPE}")

# ---------------------------------
# cmake-format: off
# ---------------------------------
# --- Forward variables cleanly ---
set(VARS_TO_FORWARD
  IPPL_PLATFORMS
  IPPL_OPENMP_THREADS
  IPPL_ENABLE_SCRIPTS
  Heffte_VERSION
  Kokkos_VERSION
  MPIEXEC_EXECUTABLE
  MPIEXEC_PREFLAGS      
  MPIEXEC_MAX_NUMPROCS
)

foreach(VAR IN LISTS VARS_TO_FORWARD)
  if(DEFINED ${VAR})
    set(VAL "${${VAR}}")
    
    if("${VAL}" MATCHES ";")
      # 1. Force CMake to treat it as a literal string cache entry to preserve semicolons
      string(APPEND CTEST_CONFIGURE_COMMAND " -D${VAR}:STRING=${VAL}")
      
      # 2. Expose it to the local CTest script scope so CTest's test launcher 
      # can parse it as a native list during the execution phase.
      set(${VAR} "${VAL}")
    else()
      # Standard scalar variable (no semicolons)
      string(APPEND CTEST_CONFIGURE_COMMAND " -D${VAR}=${VAL}")
    endif()
  endif()
endforeach()
# ---------------------------------
# cmake-format: on
# ---------------------------------

# --- Append remaining static flags ---
string(APPEND CTEST_CONFIGURE_COMMAND " -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON")
string(APPEND CTEST_CONFIGURE_COMMAND " -DIPPL_ENABLE_SOLVERS=ON")
string(APPEND CTEST_CONFIGURE_COMMAND " -DIPPL_MARK_FAILING_TESTS=ON")

if(DEFINED Kokkos_ARCH_FLAG)
  string(APPEND CTEST_CONFIGURE_COMMAND " -D${Kokkos_ARCH_FLAG}=ON")
endif()

# --- Output our configure command for debug purposes---
message("Final CTest configure command: ${CTEST_CONFIGURE_COMMAND}")

# --- configure & build ---
ctest_configure(RETURN_VALUE configure_result)
ctest_build(RETURN_VALUE build_result)

# --- fail if any test failed ---
# note that we don't submit, if all is ok, because the test phase will submit anyway
if(configure_result OR build_result)
  # submit configure + build results
  ctest_submit()
  # make sure to fail the build if configure or build failed
  message(FATAL_ERROR "CTest reported configure/build failures")
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
