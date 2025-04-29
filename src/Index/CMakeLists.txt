# -----------------------------------------------------------------------------
# src/Index/CMakeLists.txt
#
# Adds multi-dimensional index headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install Index-related headers
install(FILES
    Index.h
    Index.hpp
    NDIndex.h
    NDIndex.hpp
    DESTINATION include/Index
)

