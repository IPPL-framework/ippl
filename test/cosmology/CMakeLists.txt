file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding Hermiticity test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src          # core headers
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..                 # Kokkos libs picked up like the other tests
)

# main IPPL static / shared library (already built in top level)
set (IPPL_LIBS ippl)

add_executable (TestHermiticity TestHermiticity.cpp)

# give the test-only accessor to the compiler
target_compile_definitions (TestHermiticity PRIVATE IPPL_ENABLE_TESTS)

target_link_libraries (
    TestHermiticity
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)
