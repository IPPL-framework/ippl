# -----------------------------------------------------------------------------
# src/Stream/CMakeLists.txt
# -----------------------------------------------------------------------------
add_subdirectory(InSitu)
add_subdirectory(Registry)

if(IPPL_ENABLE_CONDUIT_IO)
    add_subdirectory(IO)
endif()