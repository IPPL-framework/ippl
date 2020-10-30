file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_LIBRARY_DIRS}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (TestPRegion TestPRegion.cpp)
target_link_libraries (
    TestPRegion
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
    boost_timer
)

add_executable (TestNDRegion TestNDRegion.cpp)
target_link_libraries (
    TestNDRegion
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
