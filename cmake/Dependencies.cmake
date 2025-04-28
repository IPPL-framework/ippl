# -----------------------------------------------------------------------------
# Dependencies.cmake
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
# -----------------------------------------------------------------------------
include(FetchContent)

# ------------------------------------------------------------------------------
# MPI
# ------------------------------------------------------------------------------
find_package(MPI COMPONENTS CXX REQUIRED)
colour_message(STATUS ${Green} "✅ MPI found ${MPI_CXX_VERSION}")

# ------------------------------------------------------------------------------
# CUDA
# ------------------------------------------------------------------------------
if("CUDA" IN_LIST IPPL_PLATFORMS)
  find_package(CUDAToolkit REQUIRED)
  colour_message(STATUS ${Green} "✅ CUDA platform requested and CUDAToolkit found ${CUDAToolkit_VERSION}")
endif()

# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------
if("OPENMP" IN_LIST IPPL_PLATFORMS)
  find_package(OpenMP REQUIRED)
  colour_message(STATUS ${Green} "✅ OpenMP platform requested OpenMP found ${OPENMP_VERSION}")
endif()

# ------------------------------------------------------------------------------
# Utility function to clear a list of vars one by one
# ------------------------------------------------------------------------------
function(unset_vars)
  foreach(VAR IN LISTS ARGN)
    unset(${VAR} PARENT_SCOPE)
  endforeach()
endfunction()

# ------------------------------------------------------------------------------
# Utility function to get git tag/sha/version from version string
# ------------------------------------------------------------------------------
function(extract_git_label VERSION_STRING RESULT_VAR)
  if("${${VERSION_STRING}}" MATCHES "^git\\.(.+)$")
    set(${RESULT_VAR} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  else()
    unset(${RESULT_VAR} PARENT_SCOPE)
  endif()
endfunction()

# ------------------------------------------------------------------------------
# Kokkos
# Use find_package with mininimum version requested,
# if that's not found, or not ok (has wrong backends etc), then build from source
# ------------------------------------------------------------------------------
# set the default version of kokkos we will ask for if not already set
if (NOT Kokkos_VERSION_DEFAULT)
  set(Kokkos_VERSION_DEFAULT 4.5.00)
endif()
# if the user has not asked for a specific version, we will use the default
if (NOT Kokkos_VERSION)
  set(Kokkos_VERSION ${Kokkos_VERSION_DEFAULT})
endif()

# is Kokkos_VERSION a git tag/branch/sha
extract_git_label(Kokkos_VERSION KOKKOS_VERSION_GIT)
if (KOKKOS_VERSION_GIT)
  # the user has asked for a particular version built from source
  set(kokkos_fetch
    GIT_TAG ${KOKKOS_VERSION_GIT}
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git)
else()
  # the user has asked for a version - use find or checkout if needed
  set(kokkos_fetch
    GIT_TAG ${Kokkos_VERSION}
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    FIND_PACKAGE_ARGS ${Kokkos_VERSION} COMPONENTS ${IPPL_PLATFORMS}
  )
endif()

# Invoke cmake fetch/find
colour_message(STATUS ${Green} "Fetching Kokkos : ${kokkos_fetch}")
FetchContent_Declare(Kokkos ${kokkos_fetch})
FetchContent_MakeAvailable(Kokkos)

# Check that kokkos actually has the platform backends that we need
if (Kokkos_FOUND)
  set(KOKKOS_PLATFORM_OK TRUE)
  foreach(PLATFORM ${IPPL_PLATFORMS})
    kokkos_check(DEVICES "${PLATFORM}" RETURN_VALUE PLATFORM_OK)
    if (NOT PLATFORM_OK)
      colour_message(FATAL_ERROR ${Red}
        "❌ Kokkos does not have backend:${PLATFORM}, but IPPL requested it, use -DKokkos_VERSION=git.xxx (tag/branch/sha)")
    else()
      colour_message(STATUS ${Green} "✅ Kokkos has backend: ${PLATFORM}")
    endif()
  endforeach()
endif()

# ------------------------------------------------------------------------------
# Heffte (only if FFT enabled)
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_FFT)
  add_compile_definitions(IPPL_ENABLE_FFT)
  option (Heffte_ENABLE_GPU_AWARE_MPI "Is an issue ... " OFF)

  # set the default version of Heffte we will ask for if not already set
  if (NOT Heffte_VERSION_DEFAULT)
    set(Heffte_VERSION_DEFAULT 2.4.0)
  endif()
  # if the user has not asked for a particular version, we will use the default
  if (NOT Heffte_VERSION)
    # default is "git.9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda",
    # spack installed heffte@git.9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda returns 2.4.0
    set(Heffte_VERSION ${Heffte_VERSION_DEFAULT})
  endif()

  # is Heffte_VERSION a git tag/branch/sha
  extract_git_label(Heffte_VERSION HEFFTE_VERSION_GIT)
  if (HEFFTE_VERSION_GIT)
    # the user has asked for a particular version built from source
    set(heffte_fetch
      GIT_TAG ${HEFFTE_VERSION_GIT}
      GIT_REPOSITORY https://github.com/icl-utk-edu/heffte.git
      DOWNLOAD_EXTRACT_TIMESTAMP ON)
  else()
    # the user has asked for a version - use find or checkout if needed
    set(heffte_fetch
      GIT_TAG ${Heffte_VERSION}
      GIT_REPOSITORY https://github.com/icl-utk-edu/heffte.git
      FIND_PACKAGE_ARGS ${Heffte_VERSION}
    )
  endif()

  # Invoke cmake fetch/find
  colour_message(STATUS ${Green} "Fetching Heffte : ${heffte_fetch}")
  FetchContent_Declare(Heffte ${heffte_fetch})

  # If building from source: Define backend options BEFORE calling MakeAvailable
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Heffte_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
    set(Heffte_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    colour_message(STATUS ${Red} "❗ Disabling AVX2 and CUDA in Debug build")
  endif()
  if(NOT DEFINED Heffte_ENABLE_FFTW AND NOT DEFINED Heffte_ENABLE_CUDA AND NOT DEFINED Heffte_ENABLE_MKL)
    set(Heffte_ENABLE_AVX2 ON)
    set(Heffte_ENABLE_FFTW OFF)
    set(Heffte_ENABLE_CUDA OFF)
  endif()

  FetchContent_MakeAvailable(Heffte)
  if (Heffte_FOUND)
    colour_message(STATUS ${Green} "✅ Heffte FOUND ${Heffte_VERSION}")
  endif()

  if ("CUDA" IN_LIST IPPL_PLATFORMS AND NOT Heffte_ENABLE_CUDA)
    colour_message(WARNING ${Red} "Heffte NOT CUDA enabled but IPPL platform CUDA requested")
  endif()
  if (Heffte_ENABLE_FFTW AND NOT Heffte_ENABLE_FFTW)
    colour_message(WARNING ${Red} "Heffte NOT FFTW enabled but IPPL FFTW requested")
  endif()

  if(TARGET Heffte AND NOT TARGET Heffte::Heffte)
    add_library(Heffte::Heffte ALIAS Heffte)
    message(STATUS "🔗 Created ALIAS Heffte::Heffte for Heffte target.")
  endif()
endif()

# ------------------------------------------------------------------------------
# GoogleTest
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_UNIT_TESTS)
  find_package(GTest)
  if (NOT GTest_FOUND)
    FetchContent_Declare(
      GTest
      GIT_REPOSITORY "https://github.com/google/googletest"
      GIT_TAG "v1.16.0"
      GIT_SHALLOW ON)
      # For Windows: Prevent overriding the parent project's compiler/linkersettings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(GTest)
    message(STATUS "✅ GoogleTest built from source (${GTest_VERSION})")
  endif()
endif()
