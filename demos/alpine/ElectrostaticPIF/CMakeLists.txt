# -----------------------------------------------------------------------------
# demos/alpine/ElectrostaticPIF/CMakeLists.txt
#
# Alpine Electrostatic Particle-In-Fourier (PIF) examples, optionally using FINUFFT.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/alpine/ElectrostaticPIF/")

if(IPPL_ENABLE_FINUFFT)
    set(PIF_FINUFFT_LIBS finufft)
    if(TARGET cufinufft)
        list(APPEND PIF_FINUFFT_LIBS cufinufft)
    endif()
endif()

foreach(_pif_target IN ITEMS LandauDampingPIF BumponTailInstabilityPIF PenningTrapPIF)
    add_executable(${_pif_target} ${_pif_target}.cpp)
    target_link_libraries(${_pif_target} PRIVATE IPPL::ippl ${PIF_FINUFFT_LIBS})
    target_include_directories(${_pif_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# LandauDampingPIF supports both the upsampled (default) and the pruned NUFFT
# pipelines via an optional 10th positional argument ("pruned"); no separate
# executable is required.
