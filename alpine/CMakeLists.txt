# -----------------------------------------------------------------------------
# alpine/CMakeLists.txt
#
# This module provides example physics applications using IPPL.
# -----------------------------------------------------------------------------

add_subdirectory(ExamplesWithoutPicManager)

function(add_alpine_example name)
  add_executable(${name} ${name}.cpp)

  target_link_libraries(${name} PRIVATE IPPL::ippl)

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

if(IPPL_ENABLE_TESTS)
  # cmake-format: off
  add_ippl_integration_test(LandauDamping 
    ARGS 16 16 16 10000000 10 FFT 0.01 LeapFrog --overallocate 2.0 --info 10
    LABELS alpine integration)
  # cmake-format: on
else()
  add_alpine_example(LandauDamping)
endif()

add_alpine_example(PenningTrap)
add_alpine_example(BumponTailInstability)
