file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding test ippltoyfdtd found in ${_relPath}")

set (IPPLTOYFDTD_SRCS
    ToyFDTD-DataSink/DataSink.cpp
)

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ippl/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/ToyFDTD-DataSink
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_LIBRARY_DIRS}
)

set (IPPL_LIBS ippl)

add_executable (ipplToyFDTD ${IPPLTOYFDTD_SRCS} ToyFDTD-DataSink/ipplToyFDTD2.cpp)
target_link_libraries (
    ipplToyFDTD
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
