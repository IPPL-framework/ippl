cmake_minimum_required(VERSION 3.20)

project(IPPL LANGUAGES CXX C VERSION 3.0.2)

include(GNUInstallDirs)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(IPPL_ENABLE_UNIT_TESTS "Enable unit tests using GoogleTest" OFF)
option(IPPL_ENABLE_FFT "Enable FFT support" OFF)
option(IPPL_ENABLE_SOLVERS "Enable IPPL solvers" OFF)
option(IPPL_ENABLE_ALPINE "Enable building the Alpine module" OFF)
option(IPPL_ENABLE_COSMOLOGY "Enable building the Cosmology module" OFF)
option(IPPL_ENABLE_EXAMPLES "Enable building the Example module" OFF)
option(IPPL_ENABLE_TESTS "Build integration tests in test/ directory" OFF)
option(BUILD_SHARED_LIBS "Build libraries as shared" OFF)
option(IPPL_DYL "Build IPPL as a shared library (compat shim for BUILD_SHARED_LIBS)" OFF)
if(IPPL_DYL)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  message(WARNING "IPPL_DYL is deprecated; use -DBUILD_SHARED_LIBS=ON instead.")
endif()
option(IPPL_ENABLE_COVERAGE "Enable code coverage" OFF)
option(IPPL_ENABLE_NSYS_PROFILER "Enable Nvidia Nsys Profiler" OFF)
option(IPPL_ENABLE_CATALYST "Enable ParaView Catalyst" OFF)
option(IPPL_ENABLE_ASCENT "Enable AlpineDAV ASCENT" OFF)
option(IPPL_ENABLE_SANITIZER "Enable sanitizer(s)" OFF)
option(USE_ALTERNATIVE_VARIANT "Use modified variant implementation (required for CUDA 12.2 + GCC 12.3.0)" OFF)


add_compile_options(
  $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang,GNU>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>
  $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>
)

include(ProjectSetup)
include(Version)
include(Dependencies)

add_subdirectory(src)

include(CompilerOptions)
include(Platforms)

if(DEFINED FETCHCONTENT_BASE_DIR)
  add_compile_options(
    $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang,GNU>:-ffile-prefix-map=${FETCHCONTENT_BASE_DIR}=.3p>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-ffile-prefix-map=${FETCHCONTENT_BASE_DIR}=.3p>
  )
endif()

if(NOT DEFINED BUILD_TESTING)
  set(BUILD_TESTING OFF CACHE BOOL "Enable tests via CTest")
endif()

if(IPPL_ENABLE_UNIT_TESTS OR IPPL_ENABLE_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "" FORCE)
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
  include(CTest)
  if(IPPL_ENABLE_UNIT_TESTS)
    add_subdirectory(unit_tests)
  endif()
  if(IPPL_ENABLE_TESTS)
    add_subdirectory(test)
  endif()
endif()

if(IPPL_ENABLE_ALPINE)
    add_subdirectory(alpine)
endif()

if(IPPL_ENABLE_COSMOLOGY)
    add_subdirectory(cosmology)
endif()

if(IPPL_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()
