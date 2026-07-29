# -----------------------------------------------------------------------------
# demos/CMakeLists.txt
#
# Top-level entry point for all demo applications bundled with IPPL.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/")

# ------------------------------------------------------------------------------
# Include optional source directories for demos
# ------------------------------------------------------------------------------

if(IPPL_ENABLE_ALPINE)
  add_subdirectory(alpine)
endif()

if(IPPL_ENABLE_COSMOLOGY)
  add_subdirectory(cosmology)
endif()

if(IPPL_ENABLE_FEL)
  add_subdirectory(fel)
endif()

if(IPPL_ENABLE_COLLISIONS)
  add_subdirectory(collisions)
endif()