# -----------------------------------------------------------------------------
# src/Partition/CMakeLists.txt
#
# Adds domain partitioning headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install Partition-related headers
install(FILES
    Partitioner.h
    Partitioner.hpp
    DESTINATION include/Partition
)
