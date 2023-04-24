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

add_executable (TestSolverDesignMP TestSolverDesign.cpp)
target_link_libraries (
    TestSolverDesignMP
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestCGSolverMP TestCGSolver.cpp)
target_link_libraries (
    TestCGSolverMP
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

if (ENABLE_FFT)
    add_executable (TestGaussian_convergenceMP TestGaussian_convergence.cpp)
    target_link_libraries (
        TestGaussian_convergenceMP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestSphereMP TestSphere.cpp)
    target_link_libraries (
        TestSphereMP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (Budiardja_plotMP Budiardja_plot.cpp)
    target_link_libraries (
        Budiardja_plotMP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestGaussianMP TestGaussian.cpp)
    target_link_libraries (
        TestGaussianMP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestFFTPeriodicPoissonSolverMP TestFFTPeriodicPoissonSolver.cpp)
    target_link_libraries (
        TestFFTPeriodicPoissonSolverMP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestGaussian_biharmonicMP TestGaussian_biharmonic.cpp)
    target_link_libraries (
        TestGaussian_biharmonicMP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )
endif ()

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
