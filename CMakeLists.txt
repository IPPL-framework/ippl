# ------------------------------------------------------------------------------
# CMake version requirement
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)

# ------------------------------------------------------------------------------
# Policies - use the latest of everything
# ------------------------------------------------------------------------------
cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})

# but disable c++20 module scanning until we have better support for it this seems to cause some
# problems when using gcc@14
cmake_policy(SET CMP0155 OLD)

# ------------------------------------------------------------------------------
# Project declaration and version
# ------------------------------------------------------------------------------
project(IPPL VERSION 3.2 LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Module path, setup before including any cmake modules
# ------------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" "${PROJECT_SOURCE_DIR}/CMakeModules")

# ------------------------------------------------------------------------------
# Project setup
# ------------------------------------------------------------------------------
include(ProjectSetup)

# ------------------------------------------------------------------------------
# Primary IPPL options
# ------------------------------------------------------------------------------
set(IPPL_PLATFORMS "OPENMP;CUDA" CACHE STRING "Platforms to build IPPL for")
option(BUILD_SHARED_LIBS "Build IPPL as a shared library" OFF)
option(IPPL_ENABLE_UNIT_TESTS "Enable unit tests using GoogleTest" OFF)
option(IPPL_ENABLE_FFT "Enable FFT support" OFF)
option(IPPL_ENABLE_SOLVERS "Enable IPPL solvers" OFF)
option(IPPL_ENABLE_ALPINE "Enable building the Alpine module" OFF)
option(IPPL_ENABLE_COSMOLOGY "Enable building the Cosmology module" OFF)
option(IPPL_ENABLE_EXAMPLES "Enable building the Example module" OFF)
option(IPPL_ENABLE_TESTS "Build integration tests in test/ directory" OFF)
option(IPPL_ENABLE_COVERAGE "Enable code coverage" OFF)
option(IPPL_ENABLE_NSYS_PROFILER "Enable Nvidia Nsys Profiler" OFF)
option(IPPL_ENABLE_SANITIZER "Enable sanitizer(s)" OFF)
option(IPPL_USE_ALTERNATIVE_VARIANT
       "Use modified variant implementation (required for CUDA 12.2 + GCC 12.3.0)" OFF)
option(IPPL_USE_STANDARD_FOLDERS "Put all generated binaries in bin/lib folders" OFF)
option(IPPL_SKIP_FAILING_TESTS "Do not build/test tests that are currently marked as failing" OFF)
option(IPPL_ENABLE_SCRIPTS "Generate job script templates for some benchmarks/tests" OFF)

# "Build IPPL as a shared library (ON) or static library (OFF)" OFF) if(IPPL_DYL)
# set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE) message(WARNING "IPPL_DYL is deprecated; use
# -DBUILD_SHARED_LIBS=ON instead.") endif()

# ------------------------------------------------------------------------------
# Setup Output directories
# ------------------------------------------------------------------------------
if(IPPL_USE_STANDARD_FOLDERS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
      CACHE PATH "Single Directory for all Executables.")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
      CACHE PATH "Single Directory for all Libraries")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
      CACHE PATH "Single Directory for all static libraries.")
else()
  unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY CACHE)
  unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)
  unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY CACHE)
endif()

# ------------------------------------------------------------------------------
# cmake modules
# ------------------------------------------------------------------------------
# cmake supplied modules
include(GNUInstallDirs)
include(FetchContent)
include(CMakePackageConfigHelpers)
include(ExternalProject)
include(CTest)

# custom modules
include(Messages)
include(CompilerOptions)
include(Platforms)
include(Dependencies)
include(FailingTests)

# ------------------------------------------------------------------------------
# Deprecated vars
# ------------------------------------------------------------------------------
if(DEFINED USE_ALTERNATIVE_VARIANT)
  colour_message(
    WARNING ${Red}
    "USE_ALTERNATIVE_VARIANT is deprecated. Please set IPPL_USE_ALTERNATIVE_VARIANT instead.")
  set(IPPL_USE_ALTERNATIVE_VARIANT ${USE_ALTERNATIVE_VARIANT} CACHE BOOL "" FORCE)
endif()

# ------------------------------------------------------------------------------
# Debug: This tells the compiler to replace occurrences of ${<DIR>} with <x> in debug info and error
# messages.
# ------------------------------------------------------------------------------
add_compile_options(
  $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang,GNU>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>
  $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>)

if(DEFINED FETCHCONTENT_BASE_DIR)
  add_compile_options(
    $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang,GNU>:-ffile-prefix-map=${FETCHCONTENT_BASE_DIR}=.3p>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-ffile-prefix-map=${FETCHCONTENT_BASE_DIR}=.3p>
  )
endif()

# ------------------------------------------------------------------------------
# Define sources for project
# ------------------------------------------------------------------------------
add_subdirectory(src)

# ------------------------------------------------------------------------------
# Include testing related material (BUILD_TESTING is defined in CTest module)
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_UNIT_TESTS OR IPPL_ENABLE_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "" FORCE)
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
  if(IPPL_ENABLE_UNIT_TESTS)
    add_subdirectory(unit_tests)
  endif()
  if(IPPL_ENABLE_TESTS)
    add_subdirectory(test)
  endif()
endif()

# ------------------------------------------------------------------------------
# Include optional source directories
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_ALPINE)
  add_subdirectory(alpine)
endif()

if(IPPL_ENABLE_COSMOLOGY)
  add_subdirectory(cosmology)
endif()

if(IPPL_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if(IPPL_ENABLE_SCRIPTS)
  add_subdirectory(scripts)
endif()
