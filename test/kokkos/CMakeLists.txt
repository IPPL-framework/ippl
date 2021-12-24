file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (TestVectorField TestVectorField.cpp)
target_link_libraries (
    TestVectorField
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestVectorField2 TestVectorField2.cpp)
target_link_libraries (
    TestVectorField2
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestVectorField3 TestVectorField3.cpp)
target_link_libraries (
    TestVectorField3
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestVectorField4 TestVectorField4.cpp)
target_link_libraries (
    TestVectorField4
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
