file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding unit tests found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIRS}
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTEST_LIBRARY_DIRS}
    ${Kokkos_DIR}/..
)

add_executable (UniformCartesian UniformCartesian.cpp)
target_link_libraries (
    UniformCartesian
    ippl
    ${MPI_CXX_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
)
