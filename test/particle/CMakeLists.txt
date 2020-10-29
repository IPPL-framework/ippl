file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_LIBRARY_DIRS}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (TestIpplParticleBase TestIpplParticleBase.cpp)
target_link_libraries (
    TestIpplParticleBase
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (TestScatter TestScatter.cpp)
target_link_libraries (
    TestScatter
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (TestGather TestGather.cpp)
target_link_libraries (
    TestGather
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)


add_executable (TestExpression TestExpression.cpp)
target_link_libraries (
    TestExpression
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (PIC3d PIC3d.cpp)
target_link_libraries (
    PIC3d
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)


add_executable (TestParticleBConds TestParticleBConds.cpp)
target_link_libraries (
    TestParticleBConds
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)


# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
