file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding parallel tests found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/ippl/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_LIBRARY_DIRS}
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (2ddens 2ddens.cpp)
target_link_libraries (
    2ddens
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (dualmpi dualmpi.cpp)
target_link_libraries (
    dualmpi
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (reduce-1 reduce-1.cpp)
target_link_libraries (
    reduce-1
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (reduce-2 reduce-2.cpp)
target_link_libraries (
    reduce-2
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (volley volley.cpp)
target_link_libraries (
    volley
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