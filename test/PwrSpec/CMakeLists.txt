file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding ttrack test in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ippl/src
    ${CLASSIC_SOURCE_DIR}
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_LIBRARY_DIRS}
)

set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

set (IPPL_LIBS ippl)

add_executable (testPwrSpec testPwrSpec.cpp)
target_link_libraries (
    testPwrSpec
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