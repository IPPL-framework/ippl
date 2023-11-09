file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding unit tests found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${GTEST_INCLUDE_DIRS}
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTEST_LIBRARY_DIRS}
    ${Kokkos_DIR}/..
)

add_executable (EdgeElement EdgeElement.cpp)
target_link_libraries (
    EdgeElement
    ippl
    pthread
    ${MPI_CXX_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
)

add_executable (QuadrilateralElement QuadrilateralElement.cpp)
target_link_libraries (
    QuadrilateralElement
    ippl
    pthread
    ${MPI_CXX_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
)

add_executable (FiniteElementSpace FiniteElementSpace.cpp)
target_link_libraries (
    FiniteElementSpace
    ippl
    pthread
    ${MPI_CXX_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
)

add_executable (LagrangeSpace LagrangeSpace.cpp)
target_link_libraries (
    LagrangeSpace
    ippl
    pthread
    ${MPI_CXX_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
)
# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
