file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding test FFT found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (TestFFTCC TestFFTCC.cpp)
target_link_libraries (
    TestFFTCC
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestFFTRC TestFFTRC.cpp)
target_link_libraries (
    TestFFTRC
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestSine TestSine.cpp)
target_link_libraries (
    TestSine
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)
add_executable (TestCos TestCos.cpp)
target_link_libraries (
    TestCos
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestCos1 TestCos1.cpp)
target_link_libraries (
    TestCos1
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
