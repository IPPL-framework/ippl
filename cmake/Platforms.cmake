# -----------------------------------------------------------------------------
# Platforms.cmake
# ~~~
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
# ~~~
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# platforms we do support
# -----------------------------------------------------------------------------
set(IPPL_SUPPORTED_PLATFORMS "SERIAL;OPENMP;CUDA;HIP")

# === Default to SERIAL if IPPL_PLATFORMS not set ===
if(NOT IPPL_PLATFORMS)
  set(IPPL_PLATFORMS "SERIAL")
  message(STATUS "No IPPL_PLATFORMS specified — defaulting to SERIAL")
endif()

# === Normalize to uppercase ===
string(TOUPPER "${IPPL_PLATFORMS}" IPPL_PLATFORMS)

# === Declare a HIP profiler option ===
if("HIP" IN_LIST IPPL_PLATFORMS)
  option(IPPL_ENABLE_HIP_PROFILER "Enable HIP Systems Profiler" OFF)
endif()

if(NOT "SERIAL" IN_LIST IPPL_PLATFORMS AND NOT "OPENMP" IN_LIST IPPL_PLATFORMS)
  list(APPEND IPPL_PLATFORMS "SERIAL")
  message(STATUS "Appending SERIAL to IPPL_PLATFORMS as no HOST execution space set")
endif()

# -----------------------------------------------------------------------------
# Sanity check for known platforms
# -----------------------------------------------------------------------------
set(unhandled_platforms_ ${IPPL_PLATFORMS})
foreach(platform ${IPPL_SUPPORTED_PLATFORMS})
  if(platform IN_LIST IPPL_PLATFORMS)
    list(REMOVE_ITEM unhandled_platforms_ ${platform})
  endif()
endforeach()

if(NOT unhandled_platforms_ STREQUAL "")
  message(FATAL_ERROR "Unknown or unsupported IPPL_PLATFORMS requested: '${unhandled_platforms_}'")
endif()

if("HIP" IN_LIST IPPL_PLATFORMS AND "CUDA" IN_LIST IPPL_PLATFORMS)
  message(FATAL_ERROR "CUDA and HIP should not both be present in IPPL_PLATFORMS")
endif()

# -----------------------------------------------------------------------------
# Profiler section
# -----------------------------------------------------------------------------
if(IPPL_ENABLE_HIP_PROFILER)
  if("HIP" IN_LIST IPPL_PLATFORMS)
    message(STATUS "🧩 Enabling HIP Profiler and KOKKOS profiliing")
    add_compile_definitions(-DIPPL_ENABLE_HIP_PROFILER)
  else()
    message(FATAL_ERROR "Cannot enable HIP Systems Profiler since platform is not HIP")
  endif()
endif()

if(IPPL_ENABLE_NSYS_PROFILER)
  if("CUDA" IN_LIST IPPL_PLATFORMS)
    message(STATUS "Enabling Nsys Profiler")
    add_compile_definitions(-DIPPL_ENABLE_NSYS_PROFILER)
  else()
    message(FATAL_ERROR "Cannot enable Nvidia Nsys Profiler since platform is not CUDA")
  endif()
endif()
