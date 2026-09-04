# -----------------------------------------------------------------------------
# SetupAdios2.cmake
#
# Records ADIOS2 / AdiosCatalyst paths for the ipplADIOS track. Resolution
# (find vs FetchContent) happens in Dependencies.cmake.
#
# Does NOT link adios2::adios2 into libippl — AdiosCatalyst is loaded at runtime
# via CATALYST_IMPLEMENTATION_*.
# -----------------------------------------------------------------------------

if(NOT TARGET ippl)
  message(FATAL_ERROR "SetupAdios2.cmake must be included after the ippl target is created.")
endif()

if(NOT IPPL_ENABLE_ADIOS2)
  return()
endif()

if(NOT IPPL_ENABLE_CATALYST)
  message(
    FATAL_ERROR
      "IPPL_ENABLE_ADIOS2 requires IPPL_ENABLE_CATALYST (AdiosCatalyst is a Catalyst 2 implementation).")
endif()

if(NOT TARGET adios2::cxx_mpi AND NOT TARGET adios2::cxx)
  message(
    FATAL_ERROR
      "ADIOS2 enabled but no adios2 imported target found. "
      "Set -DADIOS2_DIR=/path/to/adios2/install or allow FetchContent.")
endif()

target_compile_definitions(ippl PUBLIC IPPL_ENABLE_ADIOS2)

# ---------------------------------------------------------------------------
# Make the runtime backend part of the normal build
# ---------------------------------------------------------------------------
# AdiosCatalyst's Catalyst-2 implementation (libcatalyst-adios.so, its AdiosCatalib, and the in-tree
# adios2 it links) is loaded at RUNTIME via CATALYST_IMPLEMENTATION_*, so nothing links it and a
# targeted build (e.g. `make AlpineSight`) would not produce it. When it is built in-tree via
# FetchContent, tie it to the ippl library so ANY build that produces ippl — `make`, a single
# example, or any future consumer — also produces the runtime backend (and transitively adios2).
# External AdiosCatalyst installs define no such target (the .so already exists) and are skipped.
if(TARGET catalyst-adios)
  add_dependencies(ippl catalyst-adios)
endif()

# ---------------------------------------------------------------------------
# AdiosCatalyst implementation directory (runtime Catalyst backend)
# ---------------------------------------------------------------------------
if(NOT IPPL_ADIOSCATALYST_IMPL_DIR AND AdiosCatalyst_DIR)
  foreach(_ippl_ac_subdir IN ITEMS lib/catalyst lib64/catalyst catalyst)
    if(EXISTS "${AdiosCatalyst_DIR}/${_ippl_ac_subdir}/libcatalyst-adios.so")
      set(IPPL_ADIOSCATALYST_IMPL_DIR "${AdiosCatalyst_DIR}/${_ippl_ac_subdir}")
      break()
    endif()
  endforeach()
endif()
if(
  IPPL_ADIOSCATALYST_IMPL_DIR
  AND NOT EXISTS "${IPPL_ADIOSCATALYST_IMPL_DIR}/libcatalyst-adios.so"
  AND EXISTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/catalyst/libcatalyst-adios.so")
  set(IPPL_ADIOSCATALYST_IMPL_DIR "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/catalyst")
endif()

message(STATUS "ADIOS2 enabled")
message(STATUS "ADIOS2 Summary:")
message(STATUS "  External install: ${ADIOS2_FOUND}")
if(DEFINED ADIOS2_VERSION)
  message(STATUS "  Version: ${ADIOS2_VERSION}")
endif()
if(ADIOS2_PACKAGE_DIR)
  message(STATUS "  CMake config dir: ${ADIOS2_PACKAGE_DIR}")
endif()
if(ADIOS2_INSTALL_PREFIX)
  message(STATUS "  Prefix: ${ADIOS2_INSTALL_PREFIX}")
endif()
if(TARGET adios2::cxx_mpi)
  message(STATUS "  Target: adios2::cxx_mpi")
elseif(TARGET adios2::cxx)
  message(STATUS "  Target: adios2::cxx")
endif()

if(IPPL_ADIOSCATALYST_IMPL_DIR)
  message(STATUS "AdiosCatalyst Summary:")
  message(STATUS "  External install: ${AdiosCatalyst_FOUND}")
  if(DEFINED AdiosCatalyst_VERSION)
    message(STATUS "  Version: ${AdiosCatalyst_VERSION}")
  endif()
  if(AdiosCatalyst_INSTALL_PREFIX)
    message(STATUS "  Prefix: ${AdiosCatalyst_INSTALL_PREFIX}")
  endif()
  message(STATUS "  Implementation dir: ${IPPL_ADIOSCATALYST_IMPL_DIR}")
  if(TARGET catalyst-adios)
    message(STATUS "  Target: catalyst-adios")
  endif()
  message(STATUS "  Runtime:")
  message(STATUS "    export CATALYST_IMPLEMENTATION_PATHS=${IPPL_ADIOSCATALYST_IMPL_DIR}")
  message(STATUS "    export CATALYST_IMPLEMENTATION_NAME=adios")
  if(NOT EXISTS "${IPPL_ADIOSCATALYST_IMPL_DIR}/libcatalyst-adios.so")
    message(
      STATUS
        "  Note: libcatalyst-adios.so is built with the project (run cmake --build before executing).")
  endif()
elseif(AdiosCatalyst_DIR)
  message(
    WARNING
      "AdiosCatalyst_DIR=${AdiosCatalyst_DIR} set but libcatalyst-adios.so was not found.")
else()
  message(STATUS "AdiosCatalyst: not resolved (set -DAdiosCatalyst_DIR=... or allow FetchContent)")
endif()

unset(_ippl_ac_subdir)
