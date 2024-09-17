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

add_executable (BufferHandler BufferHandler.cpp)
gtest_discover_tests(BufferHandler PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    BufferHandler 
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)
