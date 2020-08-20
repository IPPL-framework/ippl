file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding particle tests found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ippl/src
    ${CLASSIC_SOURCE_DIR}
    ${H5Hut_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIR}
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_LIBRARY_DIRS}
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (PIC3d PIC3d.cpp)
target_link_libraries (
    PIC3d
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (CacheTest CacheTest.cpp)
target_link_libraries (
    CacheTest
    ${IPPL_LIBS}
    ${H5Hut_LIBRARY}
    ${HDF5_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (p3m3dHeating p3m3dHeating.cpp)
target_link_libraries (
    p3m3dHeating
    ${IPPL_LIBS}
    ${H5Hut_LIBRARY}
    ${HDF5_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (p3m3d p3m3d.cpp)
target_link_libraries (
    p3m3d
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (p3m3dTwoStreamParallel p3m3dTwoStreamParallel.cpp)
target_link_libraries (
    p3m3dTwoStreamParallel
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (p3m3dRegressionTests p3m3dRegressionTests.cpp)
target_link_libraries (
    p3m3dRegressionTests
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (p3m3dMicrobunching p3m3dMicrobunching.cpp)
target_link_libraries (
    p3m3dMicrobunching
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    ${H5Hut_LIBRARY}
    ${HDF5_LIBRARIES}
    z
    boost_timer
)

add_executable (chsr-1 chsr-1.cpp)
target_link_libraries (chsr-1 ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)

add_executable (pbconds3D pbconds3D.cpp)
target_link_libraries (pbconds3D ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)

add_executable (salman-1 salman-1.cpp)
target_link_libraries (salman-1 ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)

add_executable (test-scatter-1 test-scatter-1.cpp)
target_link_libraries (test-scatter-1 ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
