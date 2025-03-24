# -----------------------------------------------------------------------------
# src/Meshes/CMakeLists.txt
#
# Adds structured grid mesh headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install Meshes headers
install(FILES
    CartesianCentering.hpp
    CartesianCentering.h
    Centering.h
    Mesh.h
    Mesh.hpp
    UniformCartesian.h
    UniformCartesian.hpp
    DESTINATION include/Meshes
)

