# -----------------------------------------------------------------------------
# Platforms.cmake
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
# -----------------------------------------------------------------------------

# === Default to SERIAL if IPPL_PLATFORMS not set ===
if(NOT IPPL_PLATFORMS)
    set(IPPL_PLATFORMS "SERIAL")
    message(STATUS "No IPPL_PLATFORMS specified — defaulting to SERIAL")
endif()

# === Normalize to uppercase ===
string(TOUPPER "${IPPL_PLATFORMS}" IPPL_PLATFORMS)

# === Handle known platforms ===
set(unhandled_platforms_ ${IPPL_PLATFORMS})
if("SERIAL" IN_LIST IPPL_PLATFORMS)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos Serial backend" FORCE)
    set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Enable AVX2 backend for Heffte" FORCE)
    list(REMOVE_ITEM unhandled_platforms_ "SERIAL")
    message(STATUS "🧩 Backend: SERIAL")
endif()

if("OPENMP" IN_LIST IPPL_PLATFORMS)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
    set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Use Heffte Stock backend with AVX2" FORCE)
    list(REMOVE_ITEM unhandled_platforms_ "OPENMP")
    message(STATUS "🧩 Backend: OPENMP")
    endif()

if("CUDA" IN_LIST IPPL_PLATFORMS)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
    set(Heffte_ENABLE_CUDA ON CACHE BOOL "Enable Heffte CUDA backend" FORCE)
    list(REMOVE_ITEM unhandled_platforms_ "CUDA")
    message(STATUS "🧩 Backend: CUDA")
    endif()

if("HIP" IN_LIST IPPL_PLATFORMS)
    set(Heffte_ENABLE_ROCM ON CACHE BOOL "Set Heffte ROCM Backend" FORCE)
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable Kokkos HIP Backend" FORCE)
    list(REMOVE_ITEM unhandled_platforms_ "HIP")
    message(STATUS "🧩 Backend: HIP")
endif()

if(NOT unhandled_platforms_ STREQUAL "")
    message(FATAL_ERROR "Unknown or unsupported IPPL_PLATFORMS: '${unhandled_platforms_}'")
endif()

if("HIP" IN_LIST IPPL_PLATFORMS AND "CUDA" IN_LIST IPPL_PLATFORMS)
    message(FATAL_ERROR "CUDA and HIP should not both be present in IPPL_PLATFORMS")
endif()

if(NOT DEFINED Heffte_ENABLE_FFTW)
    set(Heffte_ENABLE_FFTW OFF CACHE BOOL "Enable FFTW in Heffte" FORCE)
endif()

# Profiler section
option (IPPL_ENABLE_HIP_PROFILER "Enable HIP Systems Profiler" OFF)
if (IPPL_ENABLE_HIP_PROFILER)
    if ("HIP" IN_LIST IPPL_PLATFORMS)
        set(KOKKOS_ENABLE_PROFILING ON CACHE BOOL "Enable Kokkos Profiling" FORCE)
        set(Kokkos_ENABLE_LIBDL ON CACHE BOOL "Enable LIBDL" FORCE)
        message (STATUS "🧩 Enabling HIP Profiler and KOKKOS profiliing")
        add_compile_definitions(-DIPPL_ENABLE_HIP_PROFILER)
    else()
        message (FATAL_ERROR "Cannot enable HIP Systems Profiler since platform is not HIP")
    endif()
endif()

if (IPPL_ENABLE_NSYS_PROFILER)
    if ("CUDA" IN_LIST IPPL_PLATFORMS)
        message (STATUS "Enabling Nsys Profiler")
        add_compile_definitions(-DIPPL_ENABLE_NSYS_PROFILER)
    else()
        message (FATAL_ERROR "Cannot enable Nvidia Nsys Profiler since platform is not CUDA")
    endif()
endif()
