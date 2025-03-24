# -----------------------------------------------------------------------------
# src/MaxwellSolvers/CMakeLists.txt
#
# Adds Maxwell solver header to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install MaxwellSolvers header
install(FILES
    Maxwell.h
    DESTINATION include/MaxwellSolvers
)

