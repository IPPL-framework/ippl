# -----------------------------------------------------------------------------
# Platforms.cmake
# cmake-format: off
#
# Handles platform/backend selection for IPPL: SERIAL, OPENMP, CUDA, or CUDA + OPENMP.
#
# Responsibilities:
#   - Set IPPL_PLATFORMS to default (SERIAL) if unset
#   - Normalize and validate the value
#   - Enable relevant Kokkos/Heffte options
#
# Not responsible for:
#   - Selecting default CMake build type    → ProjectSetup.cmake
#
# cmake-format: on
# -----------------------------------------------------------------------------

# === Default to SERIAL if IPPL_PLATFORMS not set ===
if(NOT IPPL_PLATFORMS)
  set(IPPL_PLATFORMS "SERIAL")
  message(STATUS "No IPPL_PLATFORMS specified — defaulting to SERIAL")
endif()

# === Normalize to uppercase ===
string(TOUPPER "${IPPL_PLATFORMS}" IPPL_PLATFORMS)

# === Handle known platform combinations ===
if("${IPPL_PLATFORMS}" STREQUAL "SERIAL")
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos Serial backend" FORCE)
  set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Enable AVX2 backend for Heffte" FORCE)
  message(STATUS "🧩 Backend: SERIAL")

elseif("${IPPL_PLATFORMS}" STREQUAL "OPENMP")
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
  set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Use Heffte Stock backend with AVX2" FORCE)
  message(STATUS "🧩 Backend: OPENMP")

elseif("${IPPL_PLATFORMS}" STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
  set(Heffte_ENABLE_CUDA ON CACHE BOOL "Enable Heffte CUDA backend" FORCE)
  message(STATUS "🧩 Backend: CUDA")

elseif("${IPPL_PLATFORMS}" STREQUAL "HIP")
  set(Heffte_ENABLE_ROCM ON CACHE BOOL "Set Heffte ROCM Backend" FORCE)
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable Kokkos HIP Backend" FORCE)
  message(STATUS "🧩 Backend: HIP")

elseif("${IPPL_PLATFORMS}" STREQUAL "CUDA;OPENMP" OR "${IPPL_PLATFORMS}" STREQUAL "OPENMP;CUDA")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
  set(Heffte_ENABLE_CUDA ON CACHE BOOL "Enable Heffte CUDA backend" FORCE)
  message(STATUS "🧩 Backend: CUDA + OPENMP")

elseif("${IPPL_PLATFORMS}" STREQUAL "HIP;OPENMP" OR "${IPPL_PLATFORMS}" STREQUAL "OPENMP;HIP")
  set(Heffte_ENABLE_ROCM ON CACHE BOOL "Set Heffte ROCM Backend" FORCE)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP Backend" FORCE)
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable Kokkos HIP Backend" FORCE)
  set(KOKKOS_ENABLE_PROFILING ON CACHE BOOL "Enable Kokkos Profiling" FORCE)
  set(Kokkos_ENABLE_LIBDL ON CACHE BOOL "Enable LIBDL" FORCE)
  message(STATUS "🧩 Backend: HIP + OPENMP")

else()
  message(FATAL_ERROR "Unknown or unsupported IPPL_PLATFORMS: '${IPPL_PLATFORMS}'")
endif()

if(NOT DEFINED Heffte_ENABLE_FFTW)
  set(Heffte_ENABLE_FFTW OFF CACHE BOOL "Enable FFTW in Heffte" FORCE)
endif()

# Profiler section
option(IPPL_ENABLE_HIP_PROFILER "Enable HIP Systems Profiler" OFF)
if(IPPL_ENABLE_HIP_PROFILER)
  if("${IPPL_PLATFORMS}" STREQUAL "HIP" OR "${IPPL_PLATFORMS}" STREQUAL "HIP;OPENMP"
     OR "${IPPL_PLATFORMS}" STREQUAL "OPENMP;HIP")
    message(STATUS "Enabling HIP Profiler")
    add_compile_definitions(-DIPPL_ENABLE_HIP_PROFILER)
  else()
    message(FATAL_ERROR "Cannot enable HIP Systems Profiler since platform is not HIP")
  endif()
endif()

if(IPPL_ENABLE_NSYS_PROFILER)
  if("${IPPL_PLATFORMS}" STREQUAL "CUDA" OR "${IPPL_PLATFORMS}" STREQUAL "CUDA;OPENMP"
     OR "${IPPL_PLATFORMS}" STREQUAL "OPENMP;CUDA")
    message(STATUS "Enabling Nsys Profiler")
    add_compile_definitions(-DIPPL_ENABLE_NSYS_PROFILER)
  else()
    message(FATAL_ERROR "Cannot enable Nvidia Nsys Profiler since platform is not CUDA")
  endif()
endif()
