file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (TestScatter TestScatter.cpp)
target_link_libraries (TestScatter ${IPPL_LIBS})

add_executable (TestGather TestGather.cpp)
target_link_libraries (TestGather ${IPPL_LIBS})

add_executable (PICnd PICnd.cpp)
target_link_libraries (PICnd ${IPPL_LIBS})

add_executable (benchmarkParticleUpdate benchmarkParticleUpdate.cpp)
target_link_libraries (benchmarkParticleUpdate ${IPPL_LIBS})

add_executable (TestHashedScatter TestHashedScatter.cpp)
target_link_libraries (TestHashedScatter ${IPPL_LIBS})

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
