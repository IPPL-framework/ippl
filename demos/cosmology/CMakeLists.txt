# -----------------------------------------------------------------------------
# demos/cosmology/CMakeLists.txt
#
# Cosmology demo for large-scale structure formation simulations.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/cosmology/")

add_executable(StructureFormation
    StructureFormation.cpp
    mc-4-Initializer/Cosmology.cpp
    mc-4-Initializer/DataBase.cpp
    mc-4-Initializer/InputParser.cpp
)

target_compile_definitions(StructureFormation PRIVATE USENAMESPACE)

target_link_libraries(StructureFormation PRIVATE IPPL::ippl)
