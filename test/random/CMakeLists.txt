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

#add_executable (TestInverseTransformSampling TestInverseTransformSampling.cpp)
#target_link_libraries (
#    TestInverseTransformSampling
#    ${IPPL_LIBS}
#    ${MPI_CXX_LIBRARIES}
#)


add_executable (TestInverseTransformSampling_1D TestInverseTransformSampling_1D.cpp)

target_link_libraries (
    TestInverseTransformSampling_1D
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)


add_executable (TestInverseTransformSampling_2D TestInverseTransformSampling_2D.cpp)

target_link_libraries (
    TestInverseTransformSampling_2D
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
