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

add_executable(TestZeroBC_sin TestZeroBC_sin.cpp)
target_link_libraries(
    TestZeroBC_sin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestZeroBC_constant1d TestZeroBC_constant1d.cpp)
target_link_libraries(
    TestZeroBC_constant1d
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestZeroBC_polynomial2d TestZeroBC_polynomial2d.cpp)
target_link_libraries(
    TestZeroBC_polynomial2d
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestPeriodicBC_sin TestPeriodicBC_sin.cpp)
target_link_libraries(
    TestPeriodicBC_sin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestPeriodicBC_sinsin TestPeriodicBC_sinsin.cpp)
target_link_libraries(
    TestPeriodicBC_sinsin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestScaling_ZeroBC_sin TestScaling_ZeroBC_sin.cpp)
target_link_libraries(
    TestScaling_ZeroBC_sin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)
add_executable(TestScaling_PeriodicBC_sinsin TestScaling_PeriodicBC_sinsin.cpp)
target_link_libraries(
    TestScaling_PeriodicBC_sinsin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)


add_executable(TestFEMSolverZeroBC_sin TestFEMSolverZeroBC_sin.cpp)
target_link_libraries(
    TestFEMSolverZeroBC_sin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestFEMVectorZeroBC_sin TestFEMVectorZeroBC_sin.cpp)
target_link_libraries(
    TestFEMVectorZeroBC_sin
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable(TestMaxwellDiffusionZeroBC TestMaxwellDiffusionZeroBC.cpp)
target_link_libraries(
    TestMaxwellDiffusionZeroBC
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
