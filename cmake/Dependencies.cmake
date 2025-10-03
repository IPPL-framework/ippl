# -----------------------------------------------------------------------------
# Dependencies.cmake
# ~~~
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
# ~~~
# -----------------------------------------------------------------------------
set(FETCHCONTENT_BASE_DIR "${PROJECT_BINARY_DIR}/_deps")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON) # opt out of auto-updates
set(FETCHCONTENT_QUIET ON)

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
  colour_message(STATUS ${Green}
                 "✅ CUDA platform requested and CUDAToolkit found ${CUDAToolkit_VERSION}")
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
# Utility function to get git tags from a repo before downloading (unused currently)
# ------------------------------------------------------------------------------
function(get_git_tags GIT_REPOSITORY RESULT_VAR)
  message("Fetching git tags for repo ${GIT_REPOSITORY}")
  execute_process(
    COMMAND git -c versionsort.suffix=- ls-remote --tags --sort=v:refname ${GIT_REPOSITORY}
    COMMAND cut --delimiter=/ --fields=3
    COMMAND grep -Po "^[\\d.]+$"
    OUTPUT_VARIABLE GIT_TAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Convert the output string into a CMake list
  string(REPLACE "\n" ";" GIT_TAGS_LIST "${GIT_TAGS}")
  set(${RESULT_VAR} "${GIT_TAGS_LIST}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# ~~~
# utility function to set kokkos options
# NB. We only set these options if we are building Kokkos from source.
# ~~~
# -----------------------------------------------------------------------------
function(set_kokkos_options)
  foreach(platform ${IPPL_SUPPORTED_PLATFORMS})
    if(platform IN_LIST IPPL_PLATFORMS)
      set(Kokkos_ENABLE_${platform} ON CACHE BOOL "Enable Kokkos ${platform} backend" FORCE)
    else()
      set(Kokkos_ENABLE_${platform} OFF CACHE BOOL "Disable Kokkos ${platform} backend" FORCE)
    endif()
    message(STATUS "IPPL_PLATFORM set: Kokkos_ENABLE_${platform} '${Kokkos_ENABLE_${platform}}'")

    if(${platform} STREQUAL "HIP" AND platform IN_LIST IPPL_PLATFORMS)
      if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        message(FATAL_ERROR "HIP platform requested but CMAKE_HIP_ARCHITECTURES not set")
      endif()
    endif()
  endforeach()

  if("CUDA" IN_LIST IPPL_PLATFORMS)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON)
  endif()

  if("HIP" IN_LIST IPPL_PLATFORMS)
    if(IPPL_ENABLE_HIP_PROFILER)
      set(KOKKOS_ENABLE_PROFILING ON CACHE BOOL "Enable Kokkos Profiling" FORCE)
      set(Kokkos_ENABLE_LIBDL ON CACHE BOOL "Enable LIBDL" FORCE)
    endif()
  endif()
endfunction()

# -----------------------------------------------------------------------------
# ~~~
# utility function to set heffte options
# NB. We only set these options if we are building Heffte from source.
# ~~~
# -----------------------------------------------------------------------------
function(set_heffte_options)
  if("SERIAL" IN_LIST IPPL_PLATFORMS)
    set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Enable AVX2 backend for Heffte" FORCE)
  endif()

  if("OPENMP" IN_LIST IPPL_PLATFORMS)
    set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Use Heffte Stock backend with AVX2" FORCE)
  endif()

  if(IPPL_ENABLE_FFT)
    set(Heffte_ENABLE_FFTW ON)
    if("CUDA" IN_LIST IPPL_PLATFORMS)
      set(Heffte_ENABLE_CUDA ON CACHE BOOL "Enable Heffte CUDA backend" FORCE)
    else()
      set(Heffte_ENABLE_CUDA OFF CACHE BOOL "Disable Heffte CUDA backend" FORCE)
    endif()
  endif()

  if("HIP" IN_LIST IPPL_PLATFORMS)
    set(Heffte_ENABLE_ROCM ON CACHE BOOL "Set Heffte ROCM Backend" FORCE)
  endif()

  if(NOT DEFINED Heffte_ENABLE_FFTW AND NOT DEFINED Heffte_ENABLE_CUDA AND NOT DEFINED
                                                                           Heffte_ENABLE_MKL)
    set(Heffte_ENABLE_AVX2 ON)
    set(Heffte_ENABLE_FFTW OFF)
    set(Heffte_ENABLE_CUDA OFF)
  endif()

  if(NOT DEFINED Heffte_ENABLE_FFTW)
    set(Heffte_ENABLE_FFTW OFF CACHE BOOL "Enable FFTW in Heffte" FORCE)
  endif()

endfunction()

# ------------------------------------------------------------------------------
# cmake-format: off
# Kokkos:
# Use an installed kokkos of version Kokkos_VERSION (or better) if available, except 
#   * if the user has requested a version that starts with "git.(tag/branch/sha)" 
#   * if the requested version is not found on the system
#   * if the requested version is found but doesn't have the backends/platforms we need
# then build from source. 
# We use FIND_PACKAGE_ARGS (cmake 3.24+) to allow FetchContent to find a system version
# cmake-format: on
# ------------------------------------------------------------------------------
# set the default version of kokkos we will ask for if not already set
if(NOT Kokkos_VERSION_DEFAULT)
  set(Kokkos_VERSION_DEFAULT 4.7.01)
endif()
# if the user has not asked for a specific version, we will use a default
if(NOT Kokkos_VERSION)
  set(Kokkos_VERSION ${Kokkos_VERSION_DEFAULT})
endif()

# does version use git tag/branch/sha syntax
extract_git_label(Kokkos_VERSION KOKKOS_VERSION_GIT)
set(Kokkos_REPOSITORY "https://github.com/kokkos/kokkos.git")
if(NOT KOKKOS_VERSION_GIT)
  find_package(Kokkos ${Kokkos_VERSION} QUIET COMPONENTS ${IPPL_PLATFORMS})
endif()

# If Kokkos found on system, Check that it has the platform backends that we need
if(Kokkos_FOUND)
  colour_message(STATUS ${Green} "✅ Kokkos ${Kokkos_VERSION} found externally")
  set(KOKKOS_PLATFORM_OK TRUE)
  foreach(PLATFORM ${IPPL_PLATFORMS})
    kokkos_check(DEVICES "${PLATFORM}" RETURN_VALUE PLATFORM_OK)
    if(NOT PLATFORM_OK)
      colour_message(
        FATAL_ERROR ${Red} "❌ Kokkos does not have backend:${PLATFORM}, but IPPL requested it\n"
        "use -DKokkos_VERSION=git.xxx (tag/branch/sha) (eg -DKokkos_VERSION=git.4.7.01)")
    else()
      colour_message(STATUS ${Green} "✅ Kokkos has backend: ${PLATFORM}")
    endif()
  endforeach()
else()
  # Define options BEFORE calling MakeAvailable
  colour_message(STATUS ${Green} "✅ Kokkos ${KOKKOS_VERSION_GIT} building from source")
  set_kokkos_options()
  # Invoke cmake fetch/find
  FetchContent_Declare(Kokkos GIT_TAG ${KOKKOS_VERSION_GIT} GIT_REPOSITORY ${Kokkos_REPOSITORY})
  FetchContent_MakeAvailable(Kokkos)

  # get_git_tags(${Kokkos_REPOSITORY} KOKKOS_GIT_TAGS) if(NOT Kokkos_VERSION IN_LIST
  # KOKKOS_GIT_TAGS) message( FATAL_ERROR "Requested Kokkos version ${Kokkos_VERSION} not a known
  # git tag, please use one of these:\n" "${KOKKOS_GIT_TAGS}") endif()
endif()

# ------------------------------------------------------------------------------
# ~~~
# Heffte (only if FFT enabled)
# Use find_package(...) with a minimum version requested,
# except:
# - if the user has requested a version that starts with "git.(tag/branch/sha)"
# - if the requested version is not found on the system
# - if the requested version is found but doesn't have the features we requested
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_FFT)
  add_compile_definitions(IPPL_ENABLE_FFT)
  option(Heffte_ENABLE_GPU_AWARE_MPI "Is an issue ... " OFF)

  # set the default version of Heffte we will ask for if not already set
  if(NOT Heffte_VERSION_DEFAULT)
    set(Heffte_VERSION_DEFAULT 2.4.0)
  endif()
  # if the user has not asked for a particular version, we will use the default
  if(NOT Heffte_VERSION)
    # default is "git.9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda", spack installed
    # heffte@git.9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda returns 2.4.0
    set(Heffte_VERSION ${Heffte_VERSION_DEFAULT})
  endif()

  # is Heffte_VERSION a git tag/branch/sha
  extract_git_label(Heffte_VERSION HEFFTE_VERSION_GIT)
  set(Heffte_REPOSITORY "https://github.com/icl-utk-edu/heffte.git")
  if(NOT HEFFTE_VERSION_GIT)
    find_package(Heffte ${Heffte_VERSION} QUIET)
  endif()

  if(Heffte_FOUND)
    colour_message(STATUS ${Green} "✅ Heffte ${Heffte_VERSION} found externally")
    set(HEFFTE_OK TRUE)
    if("CUDA" IN_LIST IPPL_PLATFORMS AND NOT Heffte_CUDA_FOUND)
      set(HEFFTE_OK FALSE)
    endif()
    if(IPPL_ENABLE_FFTW AND NOT Heffte_FFTW_FOUND)
      set(HEFFTE_OK FALSE)
    endif()
    if(NOT HEFFTE_OK)
      colour_message(
        FATAL_ERROR
        ${Red}
        "❌ Heffte found but Heffte_CUDA_FOUND=${Heffte_CUDA_FOUND} and Heffte_FFTW_FOUND=${Heffte_FFTW_FOUND}\n"
        "use -DHeffte_VERSION=git.xxx (tag/branch/sha), (eg -DHeffte_VERSION=git.v2.4.1)")
    endif()
  else()
    # Define options BEFORE calling MakeAvailable
    colour_message(STATUS ${Green} "✅ Heffte ${Heffte_VERSION} building from source")
    set_heffte_options()
    set(heffte_fetch GIT_TAG ${HEFFTE_VERSION_GIT} GIT_REPOSITORY
                     https://github.com/icl-utk-edu/heffte.git DOWNLOAD_EXTRACT_TIMESTAMP ON)
    # Invoke cmake fetch/find
    FetchContent_Declare(Heffte ${heffte_fetch})
    FetchContent_MakeAvailable(Heffte)
  endif()

  if(Heffte_ENABLE_FFTW AND NOT Heffte_ENABLE_FFTW)
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
  if(NOT GTest_FOUND)
    FetchContent_Declare(GTest GIT_REPOSITORY "https://github.com/google/googletest"
                         GIT_TAG "v1.16.0" GIT_SHALLOW ON)

    # For Windows: force shared crt, ignored on linux
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Turn off GTest install/tests in the subproject
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(BUILD_GTEST ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(GTest)
    message(STATUS "✅ GoogleTest built from source (${GTest_VERSION})")
  endif()
endif()

if(IPPL_ENABLE_TESTS)
  set(DOWNLOADED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/downloaded_headers")
  file(DOWNLOAD https://raw.githubusercontent.com/manuel5975p/stb/master/stb_image_write.h
       "${DOWNLOADED_HEADERS_DIR}/stb_image_write.h")
  message(STATUS "✅ stb_image_write loaded for testing FDTD solver.")
endif()
