cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED BUILD_TYPE)
  set(BUILD_TYPE Debug)
endif()

if(NOT DEFINED CDASH_LABEL)
  set(CDASH_LABEL "branch")
endif()

if(NOT DEFINED BUILD_DIR)
  message(FATAL_ERROR "BUILD_DIR must be defined")
endif()

if(NOT DEFINED TEST_INFO)
  set(TEST_INFO "build")
endif()

# --- CDash metadata ---
set(CTEST_SITE "${CTEST_SITE}")
set(CTEST_BUILD_CONFIGURATION ${BUILD_TYPE})
set(CTEST_BUILD_NAME "${CDASH_LABEL}-${BUILD_ARCH}-${BUILD_TYPE}-${TEST_INFO}")

set(CTEST_SOURCE_DIRECTORY "$ENV{CI_PROJECT_DIR}")
set(CTEST_BINARY_DIRECTORY "${BUILD_DIR}")
set(CTEST_CMAKE_GENERATOR "Ninja")
set(CTEST_GROUP "Experimental") # Note: This overrides your previous duplicate "Pull_Requests" line

# --- start a new build in CDash ---
ctest_start(Experimental GROUP "${CTEST_GROUP}")

# --- Initialize base configure command as a CMake LIST ---
# cmake-format: off


# --- Initialize base configure command as a flat string ---
set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND}")
string(APPEND CTEST_CONFIGURE_COMMAND " -S${CTEST_SOURCE_DIRECTORY}")
string(APPEND CTEST_CONFIGURE_COMMAND " -B${CTEST_BINARY_DIRECTORY}")
string(APPEND CTEST_CONFIGURE_COMMAND " -G${CTEST_CMAKE_GENERATOR}")
string(APPEND CTEST_CONFIGURE_COMMAND " --preset=${PRESET}")
string(APPEND CTEST_CONFIGURE_COMMAND " -DCMAKE_BUILD_TYPE=${BUILD_TYPE}")


# --- Forward variables cleanly ---
set(VARS_TO_FORWARD
  IPPL_PLATFORMS
  IPPL_OPENMP_THREADS
  IPPL_ENABLE_SCRIPTS
  Heffte_VERSION
  Kokkos_VERSION
  MPIEXEC_EXECUTABLE
  MPIEXEC_MAX_NUMPROCS
  # Note: MPIEXEC_PREFLAGS is removed from this list so it doesn't get string-quoted
)

foreach(VAR IN LISTS VARS_TO_FORWARD)
  if(DEFINED ${VAR})
    set(VAL "${${VAR}}")
    string(APPEND CTEST_CONFIGURE_COMMAND " -D${VAR}=${VAL}")
  endif()
endforeach()

# --- HANDLE MPIEXEC_PREFLAGS SAFELY FOR CTEST ---
if(DEFINED MPIEXEC_PREFLAGS)
  # 1. Propagate it to the underlying CMake configure process via a specialized cache variable syntax
  # This avoids the command-line quoting bugs entirely.
  string(APPEND CTEST_CONFIGURE_COMMAND " -DMPIEXEC_PREFLAGS:STRING=${MPIEXEC_PREFLAGS}")

  # 2. Make it a true CMake list in the current CTest script scope so CTest's 
  # test launcher knows how to separate the arguments properly.
  set(MPIEXEC_PREFLAGS "${MPIEXEC_PREFLAGS}") 
endif()

# --- Append remaining static flags ---
string(APPEND CTEST_CONFIGURE_COMMAND " -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON")
string(APPEND CTEST_CONFIGURE_COMMAND " -DIPPL_ENABLE_SOLVERS=ON")
string(APPEND CTEST_CONFIGURE_COMMAND " -DIPPL_MARK_FAILING_TESTS=ON")

if(DEFINED Kokkos_ARCH_FLAG)
  string(APPEND CTEST_CONFIGURE_COMMAND " -D${Kokkos_ARCH_FLAG}=ON")
endif()

# cmake-format: on

# --- Convert the list into the final spaced string CTest needs ---
# Safely convert the list back into a space-separated string, properly escaping semicolons in
# arguments that are lists.
message("")
message("Final CTest configure command: ${CTEST_CONFIGURE_COMMAND}")
message("")

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
