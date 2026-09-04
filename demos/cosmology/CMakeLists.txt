# -----------------------------------------------------------------------------
# demos/cosmology/CMakeLists.txt
#
# Cosmology demo for large-scale structure formation simulations.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/cosmology/")

add_executable(StructureFormation StructureFormation.cpp)

target_link_libraries(StructureFormation PRIVATE IPPL::ippl)
