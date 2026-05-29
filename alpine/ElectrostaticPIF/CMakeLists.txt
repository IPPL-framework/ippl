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

add_executable (LandauDampingPIF LandauDampingPIF.cpp)
target_link_libraries (LandauDampingPIF PRIVATE ${IPPL_LIBS})

if(IPPL_ENABLE_FINUFFT)
    target_link_libraries(LandauDampingPIF PRIVATE cufinufft finufft)
endif()

add_executable (BumponTailInstabilityPIF BumponTailInstabilityPIF.cpp)
target_link_libraries (BumponTailInstabilityPIF PRIVATE ${IPPL_LIBS})

if(IPPL_ENABLE_FINUFFT)
    target_link_libraries(BumponTailInstabilityPIF PRIVATE cufinufft finufft)
endif()

add_executable (PenningTrapPIF PenningTrapPIF.cpp)
target_link_libraries (PenningTrapPIF PRIVATE ${IPPL_LIBS})

if(IPPL_ENABLE_FINUFFT)
    target_link_libraries(PenningTrapPIF PRIVATE cufinufft finufft)
endif()

# LandauDampingPIF supports both the upsampled (default) and the pruned NUFFT
# pipelines via an optional 10th positional argument ("pruned"); no separate
# executable is required.
# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
