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

add_executable (TestSolverDesign TestSolverDesign.cpp)
target_link_libraries (
    TestSolverDesign
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestCGSolver TestCGSolver.cpp)
target_link_libraries (
    TestCGSolver
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

if (ENABLE_FFT)
    add_executable (TestGaussian_convergence TestGaussian_convergence.cpp)
    target_link_libraries (
        TestGaussian_convergence
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestSphere TestSphere.cpp)
    target_link_libraries (
        TestSphere
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (Budiardja_plot Budiardja_plot.cpp)
    target_link_libraries (
        Budiardja_plot
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestGaussian TestGaussian.cpp)
    target_link_libraries (
        TestGaussian
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestFFTPeriodicPoissonSolver TestFFTPeriodicPoissonSolver.cpp)
    target_link_libraries (
        TestFFTPeriodicPoissonSolver
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestGaussian_biharmonic TestGaussian_biharmonic.cpp)
    target_link_libraries (
        TestGaussian_biharmonic
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestGaussian_hessian TestGaussian_hessian.cpp)
    target_link_libraries (
        TestGaussian_hessian
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )

    add_executable (TestP3MSolver TestP3MSolver.cpp)
    target_link_libraries (
        TestP3MSolver
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
