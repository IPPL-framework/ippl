file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

add_subdirectory (ExamplesWithoutPicManager)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (PenningTrap PenningTrap.cpp)
target_link_libraries (PenningTrap ${IPPL_LIBS})

add_executable (LandauDamping LandauDamping.cpp)
target_link_libraries (LandauDamping ${IPPL_LIBS})

add_executable (FreeElectronLaser FreeElectronLaser.cpp)
target_link_libraries (FreeElectronLaser ${IPPL_LIBS})

add_executable (BumponTailInstability BumponTailInstability.cpp)
target_link_libraries (BumponTailInstability ${IPPL_LIBS})

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
