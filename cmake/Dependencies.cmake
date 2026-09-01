# -----------------------------------------------------------------------------
# Dependencies.cmake
# ~~~
#
# Resolves third-party libraries: Kokkos, Heffte, Conduit, and Catalyst.
#
# Responsibilities:
#   - Fetch or find Kokkos, using version and backends from Platforms.cmake
#   - Fetch Heffte if IPPL_ENABLE_FFT is ON, using CUDA or AVX2 based on platform
#   - Fetch or find Conduit when IPPL_ENABLE_CONDUIT_IO is ON
#   - Fetch or find Catalyst when IPPL_ENABLE_CATALYST is ON
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

      if(NOT DEFINED AMDGPU_TARGETS)
        set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}"
            CACHE STRING "AMDGPU targets for ROCm packages" FORCE)
      endif()
    endif()
  endforeach()

  if("CUDA" IN_LIST IPPL_PLATFORMS)
    if(Kokkos_VERSION VERSION_LESS 5.0.0)
      set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable Kokkos CUDA lambda support" FORCE)
    endif()
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
# utility function to set CMake CUDA architectures from Kokkos CUDA arch options
# NB. Heffte relies on CMAKE_CUDA_ARCHITECTURES for its CUDA kernels.
# ~~~
# -----------------------------------------------------------------------------
function(set_cuda_architectures_from_kokkos)
  if(NOT "CUDA" IN_LIST IPPL_PLATFORMS)
    return()
  endif()

  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
     AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    return()
  endif()

  # Kokkos sets Kokkos_ARCH_<Family><CC> cache booleans for each enabled CUDA architecture.  We scan
  # all cache variables, extract the trailing digits (the compute capability) from any enabled
  # Kokkos_ARCH_* entry, and use those digits directly.  This works for any existing or future
  # Kokkos arch without an explicit mapping list.
  set(cuda_architectures)

  get_cmake_property(_cache_vars CACHE_VARIABLES)
  foreach(_var IN LISTS _cache_vars)
    if(_var MATCHES "^Kokkos_ARCH_[A-Z]+([0-9]+)$" AND ${_var})
      list(APPEND cuda_architectures ${CMAKE_MATCH_1})
    endif()
  endforeach()
  unset(_cache_vars)
  unset(_var)

  if(cuda_architectures)
    list(REMOVE_DUPLICATES cuda_architectures)
    list(SORT cuda_architectures)
    set(CMAKE_CUDA_ARCHITECTURES "${cuda_architectures}"
        CACHE STRING "CUDA architectures to compile, e.g., -DCMAKE_CUDA_ARCHITECTURES=70;72" FORCE)
    message(
      STATUS
        "IPPL_PLATFORM set: CMAKE_CUDA_ARCHITECTURES '${CMAKE_CUDA_ARCHITECTURES}' from Kokkos CUDA architecture"
    )
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
      set_cuda_architectures_from_kokkos()
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

# -----------------------------------------------------------------------------
# Conduit helpers (FetchContent builds do not export conduit:: imported targets)
# -----------------------------------------------------------------------------
function(ippl_find_hdf5_dir out_var)
  # Conduit's CMake requires HDF5_DIR (install prefix). Honor explicit user/env hints first.
  if(DEFINED HDF5_DIR AND HDF5_DIR AND EXISTS "${HDF5_DIR}")
    set(${out_var} "${HDF5_DIR}" PARENT_SCOPE)
    return()
  endif()

  if(DEFINED ENV{HDF5_ROOT} AND EXISTS "$ENV{HDF5_ROOT}")
    set(${out_var} "$ENV{HDF5_ROOT}" PARENT_SCOPE)
    return()
  endif()

  find_package(HDF5 QUIET COMPONENTS C)
  if(NOT HDF5_FOUND)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  # Derive the install prefix from the library path (handles split include/lib layouts).
  set(_hdf5_lib "")
  if(HDF5_C_LIBRARIES)
    list(GET HDF5_C_LIBRARIES 0 _hdf5_lib)
  elseif(HDF5_LIBRARIES)
    list(GET HDF5_LIBRARIES 0 _hdf5_lib)
  elseif(HDF5_LIBRARY)
    set(_hdf5_lib "${HDF5_LIBRARY}")
  endif()

  if(_hdf5_lib)
    get_filename_component(_libdir "${_hdf5_lib}" REALPATH)
    get_filename_component(_libdir "${_libdir}" DIRECTORY)
    if(_libdir MATCHES "/lib(64)?$")
      get_filename_component(_hdf5_root "${_libdir}" DIRECTORY)
      set(${out_var} "${_hdf5_root}" PARENT_SCOPE)
      return()
    endif()
  endif()

  # Fallback: infer from include dir (typical Homebrew /usr/local layouts).
  if(HDF5_C_INCLUDE_DIR)
    get_filename_component(_hdf5_root "${HDF5_C_INCLUDE_DIR}" DIRECTORY)
    if(_hdf5_root MATCHES "/include$")
      get_filename_component(_hdf5_root "${_hdf5_root}" DIRECTORY)
    endif()
    set(${out_var} "${_hdf5_root}" PARENT_SCOPE)
    return()
  endif()

  set(${out_var} "" PARENT_SCOPE)
endfunction()

function(ippl_register_conduit_targets)
  if(NOT TARGET conduit)
    message(FATAL_ERROR "ippl_register_conduit_targets: expected in-tree Conduit target 'conduit'")
  endif()

  if(NOT TARGET ippl_conduit_bundle)
    add_library(ippl_conduit_bundle INTERFACE)
    target_link_libraries(ippl_conduit_bundle INTERFACE conduit conduit_relay conduit_blueprint)
    add_library(conduit::conduit ALIAS ippl_conduit_bundle)
  endif()

  # In-tree Conduit keeps C API headers under src/libs/*/c; installed layout uses include/conduit/.
  target_include_directories(
    ippl_conduit_bundle
    INTERFACE
      $<BUILD_INTERFACE:${conduit_SOURCE_DIR}/src/libs/conduit/c>
      $<BUILD_INTERFACE:${conduit_SOURCE_DIR}/src/libs/relay/c>
      $<BUILD_INTERFACE:${conduit_SOURCE_DIR}/src/libs/blueprint/c>
      $<BUILD_INTERFACE:${conduit_BINARY_DIR}/libs/conduit>
      $<BUILD_INTERFACE:${conduit_BINARY_DIR}/libs/relay>
      $<BUILD_INTERFACE:${conduit_BINARY_DIR}/libs/blueprint>)

  if(TARGET conduit_relay_mpi_io AND NOT TARGET ippl_conduit_mpi_bundle)
    add_library(ippl_conduit_mpi_bundle INTERFACE)
    target_link_libraries(
      ippl_conduit_mpi_bundle
      INTERFACE
        conduit::conduit
        conduit_relay_mpi
        conduit_relay_mpi_io
        conduit_blueprint_mpi)
    add_library(conduit::conduit_mpi ALIAS ippl_conduit_mpi_bundle)
    set(CONDUIT_RELAY_MPI_ENABLED
        TRUE
        PARENT_SCOPE)
  else()
    set(CONDUIT_RELAY_MPI_ENABLED
        FALSE
        PARENT_SCOPE)
  endif()

  set(CONDUIT_RELAY_HDF5_ENABLED
      TRUE
      PARENT_SCOPE)
endfunction()

# Conduit's BLT (and optionally Catalyst) set global CMAKE output directory cache entries
# pointing at their FetchContent build trees; restore ippl's layout before add_subdirectory(src).
function(ippl_restore_output_directories)
  if(IPPL_USE_STANDARD_FOLDERS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
        CACHE PATH "Single Directory for all Executables." FORCE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
        CACHE PATH "Single Directory for all Libraries" FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
        CACHE PATH "Single Directory for all static libraries." FORCE)
  else()
    unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY CACHE)
    unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)
    unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY CACHE)
    unset(LIBRARY_OUTPUT_PATH CACHE)
    unset(EXECUTABLE_OUTPUT_PATH CACHE)
    unset(CMAKE_Fortran_MODULE_DIRECTORY CACHE)
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
# cmake-format: on
# ------------------------------------------------------------------------------
# set the default version of kokkos we will ask for if not already set
if(NOT Kokkos_VERSION_DEFAULT)
  set(Kokkos_VERSION_DEFAULT 5.2.0)
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
  # Plain Kokkos versions are also release tags; use them for the source fallback.
  set(KOKKOS_VERSION_GIT "${Kokkos_VERSION}")
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

  # Kokkos's own CMake (kokkos_compiler_id.cmake and kokkos_functions.cmake) calls
  # find_program(Kokkos_COMPILE_LAUNCHER) and find_program(Kokkos_NVCC_WRAPPER) with HINTS/PATHS =
  # ${PROJECT_SOURCE_DIR}.  Because Kokkos never calls project() when built as a subdirectory,
  # PROJECT_SOURCE_DIR is the IPPL top-level rather than Kokkos's source dir, so the search falls
  # through to the system PATH and can pick up an unrelated installed Kokkos (e.g. a spack view of a
  # different version) - leading to the wrong kokkos_launch_compiler/nvcc_wrapper being used to
  # compile everything.
  #
  # Pin both scripts to the FetchContent source bin/ directory.  The download step of
  # FetchContent_MakeAvailable runs before add_subdirectory (which is when Kokkos's find_program
  # executes), so the scripts are guaranteed to exist by then - even on a first configure.
  set(_kokkos_fetch_bin "${FETCHCONTENT_BASE_DIR}/kokkos-src/bin")
  if(NOT Kokkos_COMPILE_LAUNCHER STREQUAL "${_kokkos_fetch_bin}/kokkos_launch_compiler")
    set(Kokkos_COMPILE_LAUNCHER "${_kokkos_fetch_bin}/kokkos_launch_compiler"
        CACHE FILEPATH "kokkos_launch_compiler pinned to FetchContent source" FORCE)
  endif()
  if(NOT Kokkos_NVCC_WRAPPER STREQUAL "${_kokkos_fetch_bin}/nvcc_wrapper")
    set(Kokkos_NVCC_WRAPPER "${_kokkos_fetch_bin}/nvcc_wrapper"
        CACHE FILEPATH "nvcc_wrapper pinned to FetchContent source" FORCE)
  endif()
  unset(_kokkos_fetch_bin)

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
if(IPPL_ENABLE_CUFFTMP AND NOT IPPL_ENABLE_FFT)
  message(FATAL_ERROR "IPPL_ENABLE_CUFFTMP requires IPPL_ENABLE_FFT=ON")
endif()

if(IPPL_ENABLE_FINUFFT AND NOT IPPL_ENABLE_FFT)
  message(FATAL_ERROR "IPPL_ENABLE_FINUFFT requires IPPL_ENABLE_FFT=ON")
endif()

if(IPPL_ENABLE_CUFINUFFT AND NOT IPPL_ENABLE_FINUFFT)
  message(FATAL_ERROR "IPPL_ENABLE_CUFINUFFT requires IPPL_ENABLE_FINUFFT=ON")
endif()

if(IPPL_ENABLE_CUFINUFFT)
  if(NOT "CUDA" IN_LIST IPPL_PLATFORMS)
    message(
      FATAL_ERROR
        "IPPL_ENABLE_CUFINUFFT requires CUDA in IPPL_PLATFORMS. "
        "cuFINUFFT is a single-rank GPU backend and this PR does not provide "
        "a host-serial to CUDA staging backend.")
  endif()
endif()

if(IPPL_ENABLE_FFT)
  add_compile_definitions(IPPL_ENABLE_FFT)

  if(IPPL_ENABLE_CUFFTMP)
    if(NOT "CUDA" IN_LIST IPPL_PLATFORMS)
      message(FATAL_ERROR "IPPL_ENABLE_CUFFTMP requires CUDA in IPPL_PLATFORMS")
    endif()

    add_compile_definitions(IPPL_ENABLE_CUFFTMP)
  endif()

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
    # if we want cuda, but the installed heffte isn't using cuda
    if("CUDA" IN_LIST IPPL_PLATFORMS AND NOT Heffte_CUDA_FOUND)
      set(HEFFTE_OK FALSE)
      set(heffte_error "CUDA support requested, but installed Heffte does not use cuda")
    endif()

    # if we want do not want cuda, but the installed heffte is using  using cuda
    if((NOT "CUDA" IN_LIST IPPL_PLATFORMS) AND Heffte_CUDA_FOUND)
      set(HEFFTE_OK FALSE)
      set(heffte_error "Installed Heffte uses CUDA but user has not requested CUDA support")
    endif()
    # if Heffte isn't using fftw
    if(IPPL_ENABLE_FFTW AND NOT Heffte_FFTW_FOUND)
      set(HEFFTE_OK FALSE)
      set(heffte_error "Installed Heffte library does not use fftw, but fft support requested")
    endif()

    if(NOT HEFFTE_OK)
      colour_message(
        FATAL_ERROR
        ${Red}
        "❌ ${heffte_error}\n"
        "IPPL_PLATFORMS=${IPPL_PLATFORMS}, Heffte_CUDA_FOUND=${Heffte_CUDA_FOUND}, Heffte_FFTW_FOUND=${Heffte_FFTW_FOUND}\n"
        "Try -DHeffte_VERSION=git.xxx (tag/branch/sha), (eg -DHeffte_VERSION=git.v2.4.1)\n")
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
# FINUFFT / cuFINUFFT (only if explicitly enabled)
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_FINUFFT)
  if(NOT FINUFFT_VERSION_DEFAULT)
    set(FINUFFT_VERSION_DEFAULT v2.5.0)
  endif()

  if(NOT FINUFFT_VERSION)
    set(FINUFFT_VERSION ${FINUFFT_VERSION_DEFAULT})
  endif()

  find_package(finufft CONFIG QUIET)
  if(NOT finufft_FOUND)
    find_package(Finufft CONFIG QUIET)
  endif()

  if(finufft_FOUND OR Finufft_FOUND)
    colour_message(STATUS ${Green} "✅ FINUFFT found externally")
  else()
    colour_message(STATUS ${Green} "✅ FINUFFT ${FINUFFT_VERSION} building from source")

    set(FINUFFT_BUILD_TESTS OFF CACHE BOOL "Disable FINUFFT tests" FORCE)
    set(FINUFFT_BUILD_EXAMPLES OFF CACHE BOOL "Disable FINUFFT examples" FORCE)
    set(FINUFFT_USE_CPU ON CACHE BOOL "Enable CPU FINUFFT" FORCE)
    set(FINUFFT_USE_DUCC0 ${IPPL_FINUFFT_USE_DUCC0}
        CACHE BOOL "Use FINUFFT's bundled DUCC0 FFT backend" FORCE)

    if(IPPL_ENABLE_CUFINUFFT)
      set_cuda_architectures_from_kokkos()
      set(FINUFFT_USE_CUDA ON CACHE BOOL "Enable CUDA cuFINUFFT" FORCE)
      colour_message(
        STATUS ${Green}
        "✅ cuFINUFFT enabled as a single-rank GPU NUFFT backend (no MPI decomposition)")
    else()
      set(FINUFFT_USE_CUDA OFF CACHE BOOL "Disable CUDA cuFINUFFT" FORCE)
    endif()

    # cuFINUFFT's CUDA RDC fatbin registration can fail at startup when its device code is wrapped
    # in a shared library. Force static for this FetchContent build regardless of the global
    # BUILD_SHARED_LIBS setting.
    set(_ippl_saved_build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)

    FetchContent_Declare(
      finufft
      GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
      GIT_TAG ${FINUFFT_VERSION}
      GIT_SHALLOW ON
      DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(finufft)

    set(BUILD_SHARED_LIBS ${_ippl_saved_build_shared_libs})
  endif()

  set(IPPL_FINUFFT_TARGETS)
  if(TARGET finufft::finufft)
    list(APPEND IPPL_FINUFFT_TARGETS finufft::finufft)
  elseif(TARGET finufft)
    list(APPEND IPPL_FINUFFT_TARGETS finufft)
  endif()

  if(IPPL_ENABLE_CUFINUFFT)
    if(TARGET cufinufft::cufinufft)
      list(APPEND IPPL_FINUFFT_TARGETS cufinufft::cufinufft)
    elseif(TARGET cufinufft)
      list(APPEND IPPL_FINUFFT_TARGETS cufinufft)
    else()
      message(FATAL_ERROR "IPPL_ENABLE_CUFINUFFT is ON, but no cuFINUFFT CMake target was found")
    endif()

    add_compile_definitions(ENABLE_GPU_NUFFT FINUFFT_USE_CUDA)
  endif()

  if(NOT IPPL_FINUFFT_TARGETS)
    message(FATAL_ERROR "IPPL_ENABLE_FINUFFT is ON, but no FINUFFT CMake target was found")
  endif()

  add_compile_definitions(ENABLE_FINUFFT)
endif()

# ------------------------------------------------------------------------------
# Conduit (Relay IO / HDF5 — separate from Catalyst's internal Conduit)
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_CONDUIT_IO)
  if(CMAKE_VERSION VERSION_LESS "3.26")
    message(
      FATAL_ERROR
        "IPPL_ENABLE_CONDUIT_IO requires CMake 3.26+ (Conduit ${Conduit_VERSION_DEFAULT}).")
  endif()

  enable_language(C)
  find_package(MPI COMPONENTS C REQUIRED)

  if(NOT Conduit_VERSION_DEFAULT)
    set(Conduit_VERSION_DEFAULT 0.9.7)
  endif()
  if(NOT Conduit_VERSION)
    set(Conduit_VERSION ${Conduit_VERSION_DEFAULT})
  endif()

  extract_git_label(Conduit_VERSION CONDUIT_VERSION_GIT)
  set(Conduit_REPOSITORY "https://github.com/LLNL/conduit.git")

  set(_conduit_hints "")
  if(DEFINED CONDUIT_DIR AND CONDUIT_DIR AND EXISTS "${CONDUIT_DIR}")
    list(APPEND _conduit_hints "${CONDUIT_DIR}" "${CONDUIT_DIR}/lib/cmake/conduit")
  endif()

  if(NOT CONDUIT_VERSION_GIT)
    find_package(Conduit ${Conduit_VERSION} QUIET HINTS ${_conduit_hints})
    set(CONDUIT_VERSION_GIT "v${Conduit_VERSION}")
  endif()

  if(Conduit_FOUND)
    colour_message(STATUS ${Green} "✅ Conduit ${Conduit_VERSION} found externally")
  else()
    colour_message(STATUS ${Green} "✅ Conduit ${CONDUIT_VERSION_GIT} building from source")

    function(set_conduit_options)
      set(ENABLE_MPI ON CACHE BOOL "" FORCE)
      set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
      set(ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
      set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
      set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(ENABLE_UTILS OFF CACHE BOOL "" FORCE)
      set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)
      set(ENABLE_RELAY_WEBSERVER OFF CACHE BOOL "" FORCE)
      set(CONDUIT_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

      ippl_find_hdf5_dir(_ippl_hdf5_dir)
      if(NOT _ippl_hdf5_dir)
        message(
          FATAL_ERROR
            "Could not find HDF5 (required for Conduit Relay IO).\n"
            "Install HDF5 development files, set -DHDF5_DIR=/path/to/hdf5/prefix, "
            "or export HDF5_ROOT.")
      endif()
      set(HDF5_DIR "${_ippl_hdf5_dir}" CACHE PATH "HDF5 location for Conduit" FORCE)
      message(STATUS "Conduit FetchContent: HDF5_DIR=${HDF5_DIR}")
    endfunction()

    set_conduit_options()
    FetchContent_Declare(
      conduit
      GIT_REPOSITORY ${Conduit_REPOSITORY}
      GIT_TAG ${CONDUIT_VERSION_GIT}
      SOURCE_SUBDIR src
      GIT_SHALLOW ON
      DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(conduit)

    ippl_register_conduit_targets()
    set(Conduit_FOUND FALSE)
    set(CONDUIT_VERSION "${Conduit_VERSION}")
    set(CONDUIT_INSTALL_PREFIX "${conduit_BINARY_DIR}")
  endif()
  unset(_conduit_hints)
endif()

# ------------------------------------------------------------------------------
# Catalyst (libcatalyst SDK — use internal Conduit to match ParaView binaries)
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_CATALYST)
  if(CMAKE_VERSION VERSION_LESS "3.26")
    message(
      FATAL_ERROR
        "IPPL_ENABLE_CATALYST requires CMake 3.26+ (Catalyst ${Catalyst_VERSION_DEFAULT}).")
  endif()

  if(NOT Catalyst_VERSION_DEFAULT)
    set(Catalyst_VERSION_DEFAULT 2.1.0)
  endif()
  if(NOT Catalyst_VERSION)
    set(Catalyst_VERSION ${Catalyst_VERSION_DEFAULT})
  endif()

  extract_git_label(Catalyst_VERSION CATALYST_VERSION_GIT)
  set(Catalyst_REPOSITORY "https://gitlab.kitware.com/paraview/catalyst.git")

  set(_catalyst_hints "")
  if(DEFINED CATALYST_HINT_PATH AND CATALYST_HINT_PATH AND EXISTS "${CATALYST_HINT_PATH}")
    list(APPEND _catalyst_hints "${CATALYST_HINT_PATH}")
  endif()

  if(NOT CATALYST_VERSION_GIT)
    find_package(catalyst ${Catalyst_VERSION} QUIET HINTS ${_catalyst_hints})
    set(CATALYST_VERSION_GIT "v${Catalyst_VERSION}")
  endif()

  if(catalyst_FOUND)
    colour_message(STATUS ${Green} "✅ Catalyst ${Catalyst_VERSION} found externally")
  else()
    colour_message(STATUS ${Green} "✅ Catalyst ${CATALYST_VERSION_GIT} building from source")

    function(set_catalyst_options)
      set(CATALYST_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
      set(CATALYST_BUILD_STUB_IMPLEMENTATION ON CACHE BOOL "" FORCE)
      set(CATALYST_BUILD_TESTING OFF CACHE BOOL "" FORCE)
      set(CATALYST_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
      set(CATALYST_WRAP_PYTHON OFF CACHE BOOL "" FORCE)
      set(CATALYST_WRAP_FORTRAN OFF CACHE BOOL "" FORCE)
      set(CATALYST_WITH_EXTERNAL_CONDUIT OFF CACHE BOOL "" FORCE)
    endfunction()

    set_catalyst_options()
    FetchContent_Declare(
      catalyst
      GIT_REPOSITORY ${Catalyst_REPOSITORY}
      GIT_TAG ${CATALYST_VERSION_GIT}
      GIT_SHALLOW ON
      DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(catalyst)

    set(catalyst_FOUND FALSE)
    if(NOT TARGET catalyst::catalyst)
      message(FATAL_ERROR "Catalyst FetchContent did not provide catalyst::catalyst")
    endif()
  endif()
  unset(_catalyst_hints)
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

# ------------------------------------------------------------------------------
# FEL module header-only dependencies (nlohmann/json for config parsing, stb_image_write for the
# Poynting-flux visualization).
# ------------------------------------------------------------------------------
if(IPPL_ENABLE_FEL)
  # Fetch the CMake package instead of downloading the release header directly.  CMake's
  # file(DOWNLOAD) does not fail by default and can leave a zero-byte json.hpp behind when a
  # release-asset host is unavailable, which only surfaces later as a confusing compile error.
  set(JSON_BuildTests OFF CACHE BOOL "Disable nlohmann/json tests" FORCE)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW ON)
  FetchContent_MakeAvailable(nlohmann_json)

  message(STATUS "✅ nlohmann/json loaded for the FEL module.")
endif()

# Restore output dirs after any FetchContent subproject (Conduit BLT is the main offender).
ippl_restore_output_directories()
