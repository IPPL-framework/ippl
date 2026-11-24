if(IPPL_ENABLE_CATALYST)
  if(NOT COMMAND catalyst_implementation)
    find_package(catalyst CONFIG REQUIRED COMPONENTS SDK)
  endif()
  # Keep the shared-library setting local to the loadable test backend.
  function(add_ippl_snapshot_capture)
    set(BUILD_SHARED_LIBS ON)
    catalyst_implementation(TARGET ippl_snapshot_capture NAME ippl_snapshot_capture
                           SOURCES SnapshotCapture.cpp)
  endfunction()
  add_ippl_snapshot_capture()

  add_ippl_test(CatalystFieldSnapshots)
  add_dependencies(CatalystFieldSnapshots ippl_snapshot_capture)
  add_ippl_test(CatalystParticleTransforms)
  add_dependencies(CatalystParticleTransforms ippl_snapshot_capture)
  set(_snapshot_tests CatalystFieldSnapshots CatalystParticleTransforms)

  if(IPPL_ENABLE_SOLVERS AND IPPL_ENABLE_FFT)
    add_ippl_test(CatalystPenningDensity
                 INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../demos/alpine")
    add_dependencies(CatalystPenningDensity ippl_snapshot_capture)
    list(APPEND _snapshot_tests CatalystPenningDensity)
  endif()

  set_tests_properties(${_snapshot_tests} PROPERTIES ENVIRONMENT
      "CATALYST_IMPLEMENTATION_NAME=ippl_snapshot_capture;CATALYST_IMPLEMENTATION_PATHS=$<TARGET_FILE_DIR:ippl_snapshot_capture>;IPPL_CATALYST_VIS=ON;IPPL_CATALYST_LIVE=OFF;IPPL_CATALYST_STEER=OFF;IPPL_CATALYST_PROXY_OPTION=OFF;IPPL_CATALYST_PNG=OFF;IPPL_CATALYST_VTK=OFF;IPPL_CATALYST_VERBOSITY=0")
endif()
