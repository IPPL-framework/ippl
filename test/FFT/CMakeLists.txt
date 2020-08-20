file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding test FFT found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ippl/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_LIBRARY_DIRS}
)

set (IPPL_LIBS ippl)

add_executable (fftspeed    fftspeed.cpp)
add_executable (TestFFT     TestFFT.cpp)
add_executable (TestFFT-1   TestFFT-1.cpp)
add_executable (TestFFT-2   TestFFT-2.cpp)
add_executable (TestFFTCos  TestFFTCos.cpp)
add_executable (TestFFTSin  TestFFTSin.cpp)
add_executable (TestFFT-XT3 TestFFT-XT3.cpp)
add_executable (TestRC      TestRC.cpp)
add_executable (TestRCMIC   TestRCMIC.cpp)

target_link_libraries (fftspeed    ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestFFT     ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestFFT-1   ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestFFT-2   ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestFFTCos  ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestFFTSin  ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestFFT-XT3 ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestRC      ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)
target_link_libraries (TestRCMIC   ${IPPL_LIBS} ${MPI_CXX_LIBRARIES} boost_timer)

add_subdirectory (SeaborgRes)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
