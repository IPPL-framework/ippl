file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
        ${CMAKE_SOURCE_DIR}/src
)

link_directories (
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl)

add_executable(P3MHeating P3MHeating.cpp)
target_link_libraries (
        P3MHeating
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
)
