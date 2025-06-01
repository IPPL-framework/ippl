# ------------------------------------------------------------------------------
# CMake version requirement
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)

# ------------------------------------------------------------------------------
# Policies - use the latest of everything
# ------------------------------------------------------------------------------
cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})

# ------------------------------------------------------------------------------
# Project declaration and version
# ------------------------------------------------------------------------------
project(IPPL VERSION 3.2 LANGUAGES CXX)

# Symbolic version string (for CLI, logs, headers, etc.)
set(IPPL_VERSION_NAME   "IPPL v${IPPL_VERSION}")

message(STATUS
    "📦 Configuring IPPL Version: ${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR} : \"${IPPL_VERSION_NAME}\""
)

# ------------------------------------------------------------------------------
# Module path
# ------------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/cmake" "${PROJECT_SOURCE_DIR}/CMakeModules")

# ------------------------------------------------------------------------------
# Primary IPPL options
# ------------------------------------------------------------------------------
option(IPPL_ENABLE_UNIT_TESTS       "Enable unit tests using GoogleTest" OFF)
option(IPPL_ENABLE_FFT              "Enable FFT support" OFF)
option(IPPL_ENABLE_SOLVERS          "Enable IPPL solvers" OFF)
option(IPPL_ENABLE_ALPINE           "Enable building the Alpine module" OFF)
option(IPPL_ENABLE_COSMOLOGY        "Enable building the Cosmology module" OFF)
option(IPPL_ENABLE_TESTS            "Build integration tests in test/ directory" OFF)
option(IPPL_DYL                     "Build IPPL as a shared library (ON) or static library (OFF)" OFF)
option(IPPL_ENABLE_COVERAGE         "Enable code coverage" OFF)
option(IPPL_ENABLE_NSYS_PROFILER    "Enable Nvidia Nsys Profiler" OFF)
option(IPPL_ENABLE_SANITIZER        "Enable sanitizer(s)" OFF)
option(IPPL_USE_ALTERNATIVE_VARIANT "Use modified variant implementation (required for CUDA 12.2 + GCC 12.3.0)" OFF)
option(IPPL_USE_STANDARD_FOLDERS    "Put all generated binaries in bin/lib folders" OFF)

# ------------------------------------------------------------------------------
# Setup Output directories
# ------------------------------------------------------------------------------
if (IPPL_USE_STANDARD_FOLDERS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin
        CACHE PATH "Single Directory for all Executables.")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/lib
        CACHE PATH "Single Directory for all Libraries")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/lib
        CACHE PATH "Single Directory for all static libraries.")
endif()

# ------------------------------------------------------------------------------
# cmake modules
# ------------------------------------------------------------------------------
# cmake supplied
include(GNUInstallDirs)
include(FetchContent)
include(CMakePackageConfigHelpers)
# custom ones
include(Messages)
include(ProjectSetup)
include(ExternalProject)
include(CompilerOptions)
include(Platforms)
include(Dependencies)

# ------------------------------------------------------------------------------
# Deprecated vars
# ------------------------------------------------------------------------------
if(DEFINED USE_ALTERNATIVE_VARIANT)
    colour_message(WARNING ${Red} "USE_ALTERNATIVE_VARIANT is deprecated. Please set IPPL_USE_ALTERNATIVE_VARIANT instead.")
    set(IPPL_USE_ALTERNATIVE_VARIANT ${USE_ALTERNATIVE_VARIANT})
endif()

# ------------------------------------------------------------------------------
# Define sources for project
# ------------------------------------------------------------------------------
add_subdirectory(src)

if (IPPL_ENABLE_UNIT_TESTS OR IPPL_ENABLE_TESTS)
    include(CTest)
    enable_testing()
endif()

if(IPPL_ENABLE_UNIT_TESTS)
    add_subdirectory(unit_tests)
endif()

if (IPPL_ENABLE_TESTS)
    add_subdirectory(test)
endif ()

if(IPPL_ENABLE_ALPINE)
    add_subdirectory(alpine)
endif()

if(IPPL_ENABLE_COSMOLOGY)
    add_subdirectory(cosmology)
endif()
