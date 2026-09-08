# Resolve host eigenanalysis independently of CUDA/HIP acceleration.
# This module is included only when IPPL_ENABLE_KOKKOS_KERNELS is enabled.
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CMakePushCheckState)

set(KokkosKernels_VERSION "5.2.0" CACHE STRING "Kokkos Kernels release or git.tag/sha")
set(IPPL_KOKKOS_KERNELS_HOST "LAPACKE" CACHE STRING "Host eigenanalysis provider: LAPACKE, MKL, NONE")
set_property(CACHE IPPL_KOKKOS_KERNELS_HOST PROPERTY STRINGS LAPACKE MKL NONE)
set(IPPL_LAPACK_INTEGER_BYTES "4" CACHE STRING "Host LAPACK integer size: 4 (LP64) or 8 (ILP64)")
set_property(CACHE IPPL_LAPACK_INTEGER_BYTES PROPERTY STRINGS 4 8)
if(NOT IPPL_KOKKOS_KERNELS_HOST MATCHES "^(LAPACKE|MKL|NONE)$")
  message(FATAL_ERROR "IPPL_KOKKOS_KERNELS_HOST must be LAPACKE, MKL, or NONE")
endif()
if(NOT IPPL_LAPACK_INTEGER_BYTES MATCHES "^(4|8)$")
  message(FATAL_ERROR "IPPL_LAPACK_INTEGER_BYTES must be 4 (LP64) or 8 (ILP64)")
endif()

extract_git_label(KokkosKernels_VERSION _ipplKernelsGit)
if(NOT TARGET Kokkos::kokkoskernels AND NOT _ipplKernelsGit)
  # Its config checks Kokkos_FOUND, whereas IPPL may have built Kokkos in-tree.
  # Keep the already selected target instead of searching for another Kokkos.
  set(_ipplKokkosFound "${Kokkos_FOUND}")
  set(Kokkos_FOUND TRUE)
  find_package(KokkosKernels ${KokkosKernels_VERSION} CONFIG QUIET)
  set(Kokkos_FOUND "${_ipplKokkosFound}")
endif()

if(NOT TARGET Kokkos::kokkoskernels)
  # These configure only source builds. Installed packages are checked below.
  set(KokkosKernels_ENABLE_TPL_LAPACKE OFF CACHE BOOL "Host LAPACKE" FORCE)
  set(KokkosKernels_ENABLE_TPL_MKL OFF CACHE BOOL "Host MKL" FORCE)
  if(IPPL_KOKKOS_KERNELS_HOST STREQUAL "LAPACKE")
    set(KokkosKernels_ENABLE_TPL_LAPACKE ON CACHE BOOL "Host LAPACKE" FORCE)
    find_path(IPPL_LAPACKE_INCLUDE_DIR lapacke.h
              HINTS ${LAPACKE_INCLUDE_DIRS} ${LAPACKE_ROOT}/include $ENV{LAPACKE_ROOT}/include)
    if(NOT IPPL_LAPACKE_INCLUDE_DIR)
      message(FATAL_ERROR "Host LAPACKE headers are required, including for CUDA/HIP. Set LAPACKE_ROOT or LAPACKE_INCLUDE_DIRS, select IPPL_KOKKOS_KERNELS_HOST=MKL, or use NONE to disable eigenanalysis.")
    endif()
    set(LAPACKE_INCLUDE_DIRS "${IPPL_LAPACKE_INCLUDE_DIR}")
    if(NOT LAPACKE_LIBRARIES)
      find_library(IPPL_LAPACKE_LIBRARY NAMES lapacke openblas
                   HINTS ${LAPACKE_LIBRARY_DIRS} ${LAPACKE_ROOT}/lib ${LAPACKE_ROOT}/lib64
                         $ENV{LAPACKE_ROOT}/lib $ENV{LAPACKE_ROOT}/lib64)
      if(NOT IPPL_LAPACKE_LIBRARY)
        message(FATAL_ERROR "Host LAPACKE library missing. Set LAPACKE_ROOT or LAPACKE_LIBRARIES (including static transitive dependencies).")
      endif()
      set(BLA_SIZEOF_INTEGER "${IPPL_LAPACK_INTEGER_BYTES}")
      find_package(LAPACK REQUIRED)
      set(LAPACKE_LIBRARIES "${IPPL_LAPACKE_LIBRARY};${LAPACK_LIBRARIES}")
    endif()
  elseif(IPPL_KOKKOS_KERNELS_HOST STREQUAL "MKL")
    set(KokkosKernels_ENABLE_TPL_MKL ON CACHE BOOL "Host MKL" FORCE)
    if(IPPL_LAPACK_INTEGER_BYTES STREQUAL "8")
      set(MKL_INTERFACE ilp64)
    else()
      set(MKL_INTERFACE lp64)
    endif()
    find_package(MKL CONFIG REQUIRED)
  endif()

  foreach(_backend CUBLAS CUSOLVER CUSPARSE ROCBLAS ROCSOLVER ROCSPARSE)
    set(_enabled OFF)
    if(("CUDA" IN_LIST IPPL_PLATFORMS AND _backend MATCHES "^CU") OR
       ("HIP" IN_LIST IPPL_PLATFORMS AND _backend MATCHES "^ROC"))
      set(_enabled ON)
    endif()
    set(KokkosKernels_ENABLE_TPL_${_backend} ${_enabled} CACHE BOOL "IPPL GPU linear algebra backend" FORCE)
  endforeach()
  # Avoid the considerable ETI build cost by instantiating only kernels used by IPPL.
  set(KokkosKernels_ADD_DEFAULT_ETI OFF CACHE BOOL "Preinstantiate default Kokkos Kernels types")
  set(KokkosKernels_ENABLE_SUPERNODAL_SPTRSV OFF CACHE BOOL "Supernodal solver requires LayoutLeft ETI")
  if(NOT _ipplKernelsGit)
    set(_ipplKernelsGit "${KokkosKernels_VERSION}")
  endif()
  FetchContent_Declare(KokkosKernels
    GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
    GIT_TAG ${_ipplKernelsGit})
  FetchContent_MakeAvailable(KokkosKernels)
  # Upstream installs an export but does not provide a build-tree export.
  export(EXPORT KokkosKernelsTargets
         FILE "${kokkoskernels_BINARY_DIR}/KokkosKernelsTargets.cmake" NAMESPACE Kokkos::)
  set(IPPL_KOKKOS_KERNELS_PACKAGE_HINT "${kokkoskernels_BINARY_DIR}")
  set(IPPL_KOKKOS_KERNELS_FETCHED ON)
else()
  set(IPPL_KOKKOS_KERNELS_PACKAGE_HINT "${KokkosKernels_DIR}")
  set(IPPL_KOKKOS_KERNELS_FETCHED OFF)
endif()

# Find the actual generated configuration header: inspect the target, not the
# user's cache options, which cannot enable features in an installed package.
get_target_property(_kernelIncludes Kokkos::kokkoskernels INTERFACE_INCLUDE_DIRECTORIES)
set(_kernelConfig)
foreach(_include IN LISTS _kernelIncludes)
  if(_include MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
    set(_include "${CMAKE_MATCH_1}")
  endif()
  if(EXISTS "${_include}/KokkosKernels_config.h")
    set(_kernelConfig "${_include}/KokkosKernels_config.h")
  endif()
endforeach()
if(NOT _kernelConfig)
  message(FATAL_ERROR "Cannot locate KokkosKernels_config.h on Kokkos::kokkoskernels")
endif()
file(READ "${_kernelConfig}" _kernelConfiguration)
foreach(_backend IN ITEMS LAPACKE MKL CUBLAS CUSOLVER CUSPARSE ROCBLAS ROCSOLVER ROCSPARSE)
  set(_has${_backend} OFF)
  if(_kernelConfiguration MATCHES "#define KOKKOSKERNELS_ENABLE_TPL_${_backend}([ \t\r\n]|$)")
    set(_has${_backend} ON)
  endif()
endforeach()
set(IPPL_KOKKOS_KERNELS_NEEDS_MKL ${_hasMKL})
if(_hasMKL)
  # Modern upstream MKL exports reference MKL::MKL, including when host
  # eigenanalysis was disabled in IPPL but the external package still uses MKL.
  find_package(MKL CONFIG REQUIRED)
endif()
foreach(_platform CUDA HIP)
  if(_platform IN_LIST IPPL_PLATFORMS)
    if((_platform STREQUAL "CUDA" AND (NOT _hasCUBLAS OR NOT _hasCUSOLVER OR NOT _hasCUSPARSE)) OR
       (_platform STREQUAL "HIP" AND (NOT _hasROCBLAS OR NOT _hasROCSOLVER OR NOT _hasROCSPARSE)))
      message(FATAL_ERROR "Selected Kokkos Kernels lacks the required ${_platform} BLAS/solver TPLs. Rebuild it with the matching backend.")
    endif()
  endif()
endforeach()

set(IPPL_KOKKOS_KERNELS_EIGENANALYSIS OFF)
if(NOT IPPL_KOKKOS_KERNELS_HOST STREQUAL "NONE")
  if(NOT _has${IPPL_KOKKOS_KERNELS_HOST})
    message(FATAL_ERROR "Selected Kokkos Kernels lacks ${IPPL_KOKKOS_KERNELS_HOST}. Rebuild it with this host TPL, or select IPPL_KOKKOS_KERNELS_HOST=NONE.")
  endif()
  set(_integerDefinition)
  if(IPPL_LAPACK_INTEGER_BYTES STREQUAL "8")
    if(IPPL_KOKKOS_KERNELS_HOST STREQUAL "MKL")
      set(_integerDefinition MKL_ILP64)
    else()
      set(_integerDefinition LAPACK_ILP64)
    endif()
  endif()
  # Export the integer interface with IPPL, also for installed-package consumers.
  set(IPPL_HOST_LAPACK_DEFINITIONS "${_integerDefinition}")
  if(IPPL_KOKKOS_KERNELS_FETCHED AND _integerDefinition)
    target_compile_definitions(kokkoskernels PUBLIC ${_integerDefinition})
  endif()

  cmake_push_check_state(RESET)
  if(IPPL_KOKKOS_KERNELS_HOST STREQUAL "MKL")
    find_package(MKL CONFIG REQUIRED)
    set(CMAKE_REQUIRED_LIBRARIES MKL::MKL)
  elseif(IPPL_KOKKOS_KERNELS_FETCHED)
    # try_compile accepts imported targets, not unbuilt local interface targets.
    add_library(ippl_lapacke_probe INTERFACE IMPORTED)
    get_target_property(_hostLinks LAPACKE INTERFACE_LINK_LIBRARIES)
    get_target_property(_hostIncludes LAPACKE INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(ippl_lapacke_probe PROPERTIES
      INTERFACE_LINK_LIBRARIES "${_hostLinks}"
      INTERFACE_INCLUDE_DIRECTORIES "${_hostIncludes}")
    set(CMAKE_REQUIRED_LIBRARIES ippl_lapacke_probe)
  else()
    set(CMAKE_REQUIRED_LIBRARIES Kokkos::kokkoskernels)
  endif()
  if(_integerDefinition)
    set(CMAKE_REQUIRED_DEFINITIONS -D${_integerDefinition})
  endif()
  set(_header lapacke.h)
  if(IPPL_KOKKOS_KERNELS_HOST STREQUAL "MKL")
    set(_header mkl.h)
  endif()
  # Recheck after provider/path changes rather than retaining a stale success.
  unset(IPPL_HOST_EIGEN_LINKS CACHE)
  set(_hostEigenProbe "
    #include <${_header}>
    #include <cmath>
    static_assert(sizeof(lapack_int) == ${IPPL_LAPACK_INTEGER_BYTES}, \"LAPACK integer ABI mismatch\");
    int main() {
      double a[4] = {0, -1, 1, 0}, er[2], ei[2], left[4], right[4];
      const lapack_int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'V', 'V', 2, a, 2, er, ei, left, 2, right, 2);
      if (info || !std::isfinite(er[0]) || !std::isfinite(ei[0]) ||
          std::abs(er[0]) > 1e-12 || std::abs(er[1]) > 1e-12 ||
          std::abs(std::abs(ei[0]) - 1) > 1e-12 || std::abs(ei[0]+ei[1]) > 1e-12) return 1;
      // Invalid layout returns -1 without touching the arrays. Check signed
      // status handling as well as a valid solve across the integer interface.
      return LAPACKE_dgeev(0, 'V', 'V', 2, a, 2, er, ei, left, 2, right, 2) != -1;
    }")
  check_cxx_source_compiles("${_hostEigenProbe}" IPPL_HOST_EIGEN_LINKS)
  if(IPPL_HOST_EIGEN_LINKS AND NOT CMAKE_CROSSCOMPILING)
    unset(IPPL_HOST_EIGEN_RUNS CACHE)
    check_cxx_source_runs("${_hostEigenProbe}" IPPL_HOST_EIGEN_RUNS)
    if(NOT IPPL_HOST_EIGEN_RUNS)
      message(FATAL_ERROR "Host LAPACKE linked but failed the eigenvalue/ABI smoke test. Check the selected libraries and integer interface.")
    endif()
  elseif(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling: host LAPACKE compile/link checked; runtime ABI validation deferred to unit tests")
  endif()
  cmake_pop_check_state()
  if(NOT IPPL_HOST_EIGEN_LINKS)
    message(FATAL_ERROR "Host LAPACKE eigenanalysis failed to compile/link. Check matching headers, LP64/ILP64 libraries, and all static dependencies (including Fortran runtime). See CMake configure log.")
  endif()
  set(IPPL_KOKKOS_KERNELS_EIGENANALYSIS ON)
endif()
message(STATUS "IPPL Kokkos Kernels: host=${IPPL_KOKKOS_KERNELS_HOST}, eigenanalysis=${IPPL_KOKKOS_KERNELS_EIGENANALYSIS}")
