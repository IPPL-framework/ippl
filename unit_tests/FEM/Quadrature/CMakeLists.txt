file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(FEMQuadratureData)
add_ippl_test(Quadrature)
add_ippl_test(GaussJacobiQuadrature)
add_ippl_test(GaussLegendreQuadrature NO_MPI)
add_ippl_test(GaussChebyshevQuadrature NO_MPI)
add_ippl_test(GaussLobattoQuadrature NO_MPI)
