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

add_executable (TestSolverDesignSP TestSolverDesign.cpp)
target_link_libraries (
    TestSolverDesignSP
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestCGSolverSP TestCGSolver.cpp)
target_link_libraries (
    TestCGSolverSP
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

if (ENABLE_FFT)
    add_executable (TestGaussian_convergenceSP TestGaussian_convergence.cpp)
    target_link_libraries (
        TestGaussian_convergenceSP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestSphereSP TestSphere.cpp)
    target_link_libraries (
        TestSphereSP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (Budiardja_plotSP Budiardja_plot.cpp)
    target_link_libraries (
        Budiardja_plotSP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestGaussianSP TestGaussian.cpp)
    target_link_libraries (
        TestGaussianSP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestFFTPeriodicPoissonSolverSP TestFFTPeriodicPoissonSolver.cpp)
    target_link_libraries (
        TestFFTPeriodicPoissonSolverSP
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestGaussian_biharmonicSP TestGaussian_biharmonic.cpp)
    target_link_libraries (
        TestGaussian_biharmonicSP
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
