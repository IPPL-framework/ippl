# -----------------------------------------------------------------------------
# Dependencies.cmake
# cmake-format: off
#
# Resolves third-party libraries: Kokkos and Heffte.
#
# Responsibilities:
#   - Fetch or find Kokkos, using version and backends from Platforms.cmake
#   - Fetch Heffte if IPPL_ENABLE_FFT is ON, using CUDA or AVX2 based on platform
#
# Not responsible for:
#   - Selecting platform backends            → Platforms.cmake
#   - Enabling compiler flags                → CompilerOptions.cmake
#   - Version variables or target creation   → Version.cmake / src/
#
# cmake-format: on
# -----------------------------------------------------------------------------
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/.fc-cache")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON) # opt out of auto-updates
set(FETCHCONTENT_QUIET ON)

include(FetchContent)

# === MPI ===
find_package(MPI REQUIRED COMPONENTS CXX)

# === Kokkos ===
set(Kokkos_VERSION "4.5.00")
message(STATUS "🔍 Looking for Kokkos ${Kokkos_VERSION}")

find_package(Kokkos ${Kokkos_VERSION} QUIET)
if(NOT Kokkos_FOUND)
  message(STATUS "📥 Kokkos not found — using FetchContent")
  FetchContent_Declare(
    kokkos URL https://github.com/kokkos/kokkos/archive/refs/tags/${Kokkos_VERSION}.tar.gz
               DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(kokkos)
endif()

message(STATUS "✅ Kokkos ready")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(Heffte_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
  message(STATUS "❗ Disabling AVX2 and CUDA in Debug build")
endif()

# === Heffte (only if FFT enabled) ===
if(IPPL_ENABLE_FFT)
  add_compile_definitions(IPPL_ENABLE_FFT)

  # Prefer a preinstalled package first
  if(NOT DEFINED Heffte_VERSION)
    set(Heffte_VERSION "master")
  endif()

  # Try to find a preinstalled Heffte first. Accept both package names: Heffte / heffte.
  set(_heffte_pkg_names Heffte heffte)
  if(Heffte_VERSION MATCHES "^[0-9]+(\\.[0-9]+)*$")
    find_package(Heffte ${Heffte_VERSION} CONFIG QUIET NAMES ${_heffte_pkg_names})
  else()
    find_package(Heffte CONFIG QUIET NAMES ${_heffte_pkg_names})
  endif()

  if(NOT Heffte_FOUND)
    message(STATUS "📦 Heffte ${Heffte_VERSION} not found — fetching")
    if(Heffte_VERSION STREQUAL "master")
      set(_heffte_repo GIT_REPOSITORY https://github.com/icl-utk-edu/heffte.git)
      set(_heffte_tag GIT_TAG 9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda)
    else()
      unset(_heffte_repo)
      set(_heffte_tag
          URL https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${Heffte_VERSION}.tar.gz)
    endif()
    set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Use AVX2 backend in Heffte" FORCE)
    set(Heffte_ENABLE_FFTW OFF CACHE BOOL "Use FFTW in Heffte" FORCE)
    set(Heffte_ENABLE_CUDA OFF CACHE BOOL "Use CUDA in Heffte" FORCE)

    FetchContent_Declare(heffte ${_heffte_repo} ${_heffte_tag} DOWNLOAD_EXTRACT_TIMESTAMP ON)

    FetchContent_MakeAvailable(heffte)

    # Some builds export 'Heffte' target instead of a namespaced one
    if(TARGET Heffte AND NOT TARGET Heffte::heffte)
      add_library(Heffte::heffte ALIAS Heffte)
      message(STATUS "🔗 Created ALIAS Heffte::heffte for Heffte target.")
    endif()

    if(NOT TARGET Heffte::heffte)
      message(FATAL_ERROR "❌ Heffte::heffte target is missing. Check Heffte build configuration.")
    endif()

    message(STATUS "✅ Heffte ready (fetched: ${Heffte_VERSION})")
  else()
    message(STATUS "✅ Found preinstalled Heffte ${Heffte_VERSION}")
  endif()
endif()

if(IPPL_ENABLE_UNIT_TESTS)
  find_package(GTest CONFIG QUIET)

  if(NOT GTest_FOUND)
    message(STATUS "📥 GoogleTest not found — fetching")

    FetchContent_Declare(
      googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
                     DOWNLOAD_EXTRACT_TIMESTAMP ON)

    # Turn off GTest install/tests in the subproject
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # harmless on non-MSVC
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(BUILD_GTEST ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  message(STATUS "✅ GoogleTest ready")
endif()
