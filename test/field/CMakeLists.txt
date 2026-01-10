file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding index test found in ${_relPath}")

add_ippl_integration_test(
  TestLaplace
  LABELS
  integration
  field
  ARGS
  10
  10
  --info
  10)

add_ippl_integration_test(TestFieldBC LABELS integration field)

add_ippl_integration_test(TestHalo LABELS integration field)

add_ippl_integration_test(
  TestCurl
  LABELS
  integration
  field
  ARGS
  10
  0
  --info
  10)

add_ippl_integration_test(
  TestHessian
  LABELS
  integration
  field
  ARGS
  10
  0
  --info
  10)

if("${Kokkos_DEVICES}" MATCHES "CUDA" AND "${Kokkos_DEVICES}" MATCHES "OPENMP")
  add_ippl_integration_test(TestMixedExec LABELS integration field)
endif()
