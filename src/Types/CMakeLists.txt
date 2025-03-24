# -----------------------------------------------------------------------------
# src/Types/CMakeLists.txt
#
# Adds core type definitions and traits to the IPPL target.
# -----------------------------------------------------------------------------

# Expose Types headers for use as <Types/...>
target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install Types-related headers
install(FILES
    Variant.h
    Vector.h
    Vector.hpp
    ViewTypes.h
    IpplTypes.h
    DESTINATION include/Types
)

