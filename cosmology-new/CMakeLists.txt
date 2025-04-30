file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding StructureFormation ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cosmology/mc-4-Initializer    
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS}) 

add_definitions(-DUSENAMESPACE)
 
# Add the main executable and link the object file
add_executable(StructureFormation StructureFormation.cpp mc-4-Initializer/Cosmology.cpp mc-4-Initializer/DataBase.cpp mc-4-Initializer/InputParser.cpp)

target_link_libraries (StructureFormation ${IPPL_LIBS})


# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
