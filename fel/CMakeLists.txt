include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)
set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (FreeElectronLaser FreeElectronLaser.cpp)
target_link_libraries (FreeElectronLaser ${IPPL_LIBS})

add_executable (TestFDTDSolverWithParticles TestFDTDSolverWithParticles.cpp)
target_link_libraries (TestFDTDSolverWithParticles ${IPPL_LIBS})