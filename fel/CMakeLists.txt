file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding FEL module found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/downloaded_headers
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (FreeElectronLaser FreeElectronLaser.cpp)
target_link_libraries (FreeElectronLaser ${IPPL_LIBS})

if (IPPL_HDF5)
    target_include_directories (FreeElectronLaser PRIVATE ${HDF5_INCLUDE_DIR})
    target_link_libraries (FreeElectronLaser ${HDF5_LIBRARIES})
    target_compile_definitions (FreeElectronLaser PRIVATE IPPL_HDF5)
endif()

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
