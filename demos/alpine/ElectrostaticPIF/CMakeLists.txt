file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
        ${CMAKE_SOURCE_DIR}/src
)

set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

if(IPPL_ENABLE_FINUFFT)
    set(PIF_FINUFFT_LIBS finufft)
    if(TARGET cufinufft)
        list(APPEND PIF_FINUFFT_LIBS cufinufft)
    endif()
endif()

foreach(_pif_target IN ITEMS LandauDampingPIF BumponTailInstabilityPIF PenningTrapPIF)
    add_executable(${_pif_target} ${_pif_target}.cpp)
    target_link_libraries(${_pif_target} PRIVATE IPPL::ippl ${PIF_FINUFFT_LIBS})
endforeach()

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
