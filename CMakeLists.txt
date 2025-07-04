cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O3 -g")

project(IPPL LANGUAGES CXX VERSION 3.0.2)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(IPPL_ENABLE_UNIT_TESTS "Enable unit tests using GoogleTest" OFF)
option(IPPL_ENABLE_FFT "Enable FFT support" OFF)
option(IPPL_ENABLE_SOLVERS "Enable IPPL solvers" OFF)
option(IPPL_ENABLE_ALPINE "Enable building the Alpine module" OFF)
option(IPPL_ENABLE_COSMOLOGY "Enable building the Cosmology module" OFF)
option(IPPL_ENABLE_EXAMPLES "Enable building the Example module" OFF)
option(IPPL_ENABLE_TESTS "Build integration tests in test/ directory" OFF)
option(IPPL_DYL "Build IPPL as a shared library (ON) or static library (OFF)" OFF)
option(IPPL_ENABLE_COVERAGE "Enable code coverage" OFF)
option(IPPL_ENABLE_NSYS_PROFILER "Enable Nvidia Nsys Profiler" OFF)
option(IPPL_ENABLE_SANITIZER "Enable sanitizer(s)" OFF)
option(USE_ALTERNATIVE_VARIANT "Use modified variant implementation (required for CUDA 12.2 + GCC 12.3.0)" OFF)


include(ProjectSetup)
include(Version)
include(CompilerOptions)
include(Platforms)
include(Dependencies)

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

if(IPPL_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()
