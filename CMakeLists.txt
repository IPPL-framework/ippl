cmake_minimum_required(VERSION 3.20)
project(IPPL LANGUAGES CXX VERSION 3.0.2)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ENABLE_UNIT_TESTS "Enable unit tests using GoogleTest" OFF)
option(ENABLE_FFT "Enable FFT support" OFF)
option (ENABLE_SOLVERS "Enable IPPL solvers" OFF)
option(ENABLE_ALPINE "Enable building the Alpine module" OFF)
option(ENABLE_TESTS "Build integration tests in test/ directory" OFF)

include(ProjectSetup)
include(Version)
include(CompilerOptions)
include(Platforms)
include(Dependencies)

add_subdirectory(src)

if (ENABLE_UNIT_TESTS OR ENABLE_TESTS)
    include(CTest)
    enable_testing()
endif()

if(ENABLE_UNIT_TESTS)
    add_subdirectory(unit_tests)
endif()

if (ENABLE_TESTS)
    add_subdirectory(test)
endif ()

if(ENABLE_ALPINE)
    add_subdirectory(alpine)
endif()
