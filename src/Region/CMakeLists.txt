# -----------------------------------------------------------------------------
# src/Region/CMakeLists.txt
#
# Adds region and region layout headers to the IPPL target.
# -----------------------------------------------------------------------------

# Expose Region headers for use as <Region/...>
target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install Region-related headers
install(FILES
    NDRegion.h
    NDRegion.hpp
    PRegion.h
    PRegion.hpp
    RegionLayout.h
    RegionLayout.hpp
    DESTINATION include/Region
)

