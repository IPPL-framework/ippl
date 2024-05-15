file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (UniformPlasmaTest UniformPlasmaTest.cpp)
target_link_libraries (UniformPlasmaTest ${IPPL_LIBS})

add_executable (LandauDampingMixedPrecision LandauDampingMixedPrecision.cpp)
target_link_libraries (LandauDampingMixedPrecision ${IPPL_LIBS})

add_executable (LandauDampingMixedExec LandauDampingMixedExec.cpp)
target_link_libraries (LandauDampingMixedExec ${IPPL_LIBS})
find_package(Threads REQUIRED)
target_link_libraries(LandauDampingMixedExec Threads::Threads)


add_executable (LandauDampingParameterList LandauDampingParameterList.cpp)
target_link_libraries (LandauDampingParameterList ${IPPL_LIBS})

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
