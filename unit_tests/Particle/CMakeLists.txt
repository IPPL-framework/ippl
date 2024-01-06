file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding unit tests found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

add_executable (ParticleBase ParticleBase.cpp)
gtest_discover_tests(ParticleBase)

target_link_libraries (
    ParticleBase
    ippl
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (ParticleBC ParticleBC.cpp)
gtest_discover_tests(ParticleBC)

target_link_libraries (
    ParticleBC
    ippl
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (ParticleSendRecv ParticleSendRecv.cpp)
gtest_discover_tests(ParticleSendRecv)

target_link_libraries (
    ParticleSendRecv
    ippl
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
