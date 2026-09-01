# -----------------------------------------------------------------------------
# SetupConduit.cmake
#
# Links Conduit Relay IO into the ippl target. Resolution (find vs FetchContent)
# happens in Dependencies.cmake.
# -----------------------------------------------------------------------------

if(NOT TARGET ippl)
  message(FATAL_ERROR "SetupConduit.cmake must be included after the ippl target is created.")
endif()

if(NOT TARGET conduit::conduit)
  message(
    FATAL_ERROR
      "Conduit IO enabled but conduit::conduit target is missing. "
      "Set -DCONDUIT_DIR=/path/to/conduit/install or allow FetchContent.")
endif()

message(STATUS "Conduit IO enabled")

if(NOT CONDUIT_RELAY_HDF5_ENABLED)
  message(
    FATAL_ERROR
      "Conduit at ${CONDUIT_INSTALL_PREFIX} was built without HDF5 Relay support.\n"
      "Rebuild Conduit with HDF5 enabled or set HDF5_DIR when using FetchContent.")
endif()

target_compile_definitions(ippl PUBLIC IPPL_ENABLE_CONDUIT_IO)

if(CONDUIT_RELAY_MPI_ENABLED)
  if(NOT TARGET conduit::conduit_mpi)
    message(FATAL_ERROR "Conduit MPI Relay is enabled but conduit::conduit_mpi was not exported.")
  endif()
  target_link_libraries(ippl PUBLIC conduit::conduit_mpi)
  target_compile_definitions(ippl PUBLIC IPPL_CONDUIT_MPI_IO)
  set(_ippl_conduit_link_target "conduit::conduit_mpi")
else()
  target_link_libraries(ippl PUBLIC conduit::conduit)
  set(_ippl_conduit_link_target "conduit::conduit")
endif()

# Compile Relay IO in a separate TU so Catalyst's bundled Conduit headers cannot
# shadow the full Conduit install used for HDF5 export.
add_library(
  ippl_conduit_relay OBJECT
  ${CMAKE_CURRENT_LIST_DIR}/../src/Stream/IO/ConduitIOWriterRelay.cpp)
target_include_directories(ippl_conduit_relay PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(ippl_conduit_relay PRIVATE ${_ippl_conduit_link_target})
if(CONDUIT_RELAY_MPI_ENABLED)
  target_compile_definitions(ippl_conduit_relay PRIVATE IPPL_CONDUIT_MPI_IO)
endif()
target_sources(ippl PRIVATE $<TARGET_OBJECTS:ippl_conduit_relay>)

# ---------------------------------------------------------------------------
# Optional: verify Catalyst + Conduit can be linked together (no symbol clash)
# ---------------------------------------------------------------------------
function(_ippl_get_imported_lib_location _target _out_var)
  if(NOT TARGET "${_target}")
    set(${_out_var} "" PARENT_SCOPE)
    return()
  endif()
  set(_loc "")
  get_target_property(_loc "${_target}" IMPORTED_LOCATION)
  if(NOT _loc)
    get_property(_cfgs TARGET "${_target}" PROPERTY IMPORTED_CONFIGURATIONS)
    foreach(_cfg IN LISTS _cfgs)
      get_target_property(_cfg_loc "${_target}" IMPORTED_LOCATION_${_cfg})
      if(_cfg_loc)
        set(_loc "${_cfg_loc}")
        break()
      endif()
    endforeach()
  endif()
  set(${_out_var} "${_loc}" PARENT_SCOPE)
endfunction()

if(IPPL_ENABLE_CATALYST AND TARGET catalyst::catalyst)
  _ippl_get_imported_lib_location(catalyst::catalyst _ippl_catalyst_lib)
  if(_ippl_catalyst_lib AND EXISTS "${_ippl_catalyst_lib}")
    find_library(
      _ippl_conduit_lib
      NAMES conduit
      HINTS "${CONDUIT_INSTALL_PREFIX}/lib" "${CONDUIT_INSTALL_PREFIX}/lib64"
      NO_DEFAULT_PATH)
    if(_ippl_conduit_lib)
      execute_process(
        COMMAND bash -c
                "comm -12 <(nm -D --defined-only '${_ippl_catalyst_lib}' 2>/dev/null | awk '{print $3}' | sort -u) <(nm -D --defined-only '${_ippl_conduit_lib}' 2>/dev/null | awk '{print $3}' | sort -u) | grep -c '^conduit_' || true"
        OUTPUT_VARIABLE _ippl_conduit_symbol_overlap
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_ippl_conduit_symbol_overlap GREATER 0)
        message(
          WARNING
            "Found ${_ippl_conduit_symbol_overlap} duplicate 'conduit_*' symbols between libcatalyst and libconduit. Linking both may be unsafe.")
      else()
        message(
          STATUS
            "Catalyst/Conduit symbol check: no duplicate conduit_* symbols (libcatalyst uses catalyst_conduit_* API)."
        )
      endif()
    endif()
  endif()
  unset(_ippl_catalyst_lib)
  unset(_ippl_conduit_lib)
  unset(_ippl_conduit_symbol_overlap)
endif()

message(STATUS "Conduit Summary:")
message(STATUS "  External install: ${Conduit_FOUND}")
message(STATUS "  Version: ${CONDUIT_VERSION}")
message(STATUS "  Prefix: ${CONDUIT_INSTALL_PREFIX}")
message(STATUS "  Link target: ${_ippl_conduit_link_target}")
message(STATUS "  CONDUIT_RELAY_HDF5_ENABLED: ${CONDUIT_RELAY_HDF5_ENABLED}")
message(STATUS "  CONDUIT_RELAY_MPI_ENABLED: ${CONDUIT_RELAY_MPI_ENABLED}")
if(CONDUIT_RELAY_ADIOS_ENABLED)
  message(STATUS "  CONDUIT_RELAY_ADIOS_ENABLED: ${CONDUIT_RELAY_ADIOS_ENABLED}")
endif()

unset(_ippl_conduit_link_target)
