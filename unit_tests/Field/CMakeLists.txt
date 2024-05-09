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

add_executable (Field Field.cpp)
gtest_discover_tests(Field PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    Field
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (FieldBC FieldBC.cpp)
gtest_discover_tests(FieldBC PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    FieldBC
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (Halo Halo.cpp)
gtest_discover_tests(Halo PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    Halo
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
