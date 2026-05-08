file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

find_package(Heffte CONFIG REQUIRED)
set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (VortexInCell VortexInCell.cpp)
add_executable(Fsl Fsl.cpp)
target_include_directories(VortexInCell PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(VortexInCell ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

target_include_directories(Fsl PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(Fsl ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)


# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
