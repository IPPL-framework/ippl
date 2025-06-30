# -----------------------------------------------------------------------------
# src/FieldLayout/CMakeLists.txt
#
# Adds FieldLayout implementation to the IPPL target.
# -----------------------------------------------------------------------------

# Register FieldLayout source
target_sources(ippl
    PRIVATE
        FieldLayout.cpp
)

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install FieldLayout headers
install(FILES
    FieldLayout.h
    FieldLayout.hpp
    DESTINATION include/FieldLayout
)

