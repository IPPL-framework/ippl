# -----------------------------------------------------------------------------
# demos/collisions/CMakeLists.txt
#
# Collisions demo using a P3M (Particle-Particle Particle-Mesh) heating simulation.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/collisions/")

add_executable(P3MHeating P3MHeating.cpp)

target_link_libraries(P3MHeating PRIVATE IPPL::ippl)
