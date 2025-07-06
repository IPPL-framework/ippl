file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(EdgeElement)
add_ippl_test(QuadrilateralElement)
add_ippl_test(HexahedralElement)
add_ippl_test(FiniteElementSpace)
add_ippl_test(LagrangeSpace)
add_ippl_test(Quadrature)
add_ippl_test(GaussJacobiQuadrature)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

add_executable (EdgeElement EdgeElement.cpp)
gtest_discover_tests(EdgeElement PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    EdgeElement
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (QuadrilateralElement QuadrilateralElement.cpp)
gtest_discover_tests(QuadrilateralElement PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    QuadrilateralElement
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable(HexahedralElement HexahedralElement.cpp)
gtest_discover_tests(HexahedralElement PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries(
    HexahedralElement
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (FiniteElementSpace FiniteElementSpace.cpp)
gtest_discover_tests(FiniteElementSpace PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    FiniteElementSpace
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (LagrangeSpace LagrangeSpace.cpp)
gtest_discover_tests(LagrangeSpace PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    LagrangeSpace
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (Quadrature Quadrature.cpp)
gtest_discover_tests(Quadrature PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    Quadrature
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable(GaussJacobiQuadrature GaussJacobiQuadrature.cpp)
gtest_discover_tests(GaussJacobiQuadrature PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries(
    GaussJacobiQuadrature
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)


add_executable(Solver Solver.cpp)
gtest_discover_tests(Solver PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries(
    Solver
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)


add_executable(NedelecSpace NedelecSpace.cpp)
gtest_discover_tests(NedelecSpace PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries(
    NedelecSpace
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable(FEMVector FEMVector.cpp)
gtest_discover_tests(FEMVector PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries(
    FEMVector
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable(LagrangeSpaceFEMVector LagrangeSpaceFEMVector.cpp)
gtest_discover_tests(LagrangeSpaceFEMVector PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries(
    LagrangeSpaceFEMVector
    ippl
    pthread
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
