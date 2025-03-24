file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

add_ippl_integration_test(TestLaplace LABELS integration field)
add_ippl_integration_test(TestFieldBC LABELS integration field)

add_ippl_integration_test(TestHalo LABELS integration field)
add_ippl_integration_test(TestCurl LABELS integration field)
add_ippl_integration_test(TestHessian LABELS integration field)


if("${Kokkos_DEVICES}" MATCHES "CUDA" AND "${Kokkos_DEVICES}" MATCHES "OPENMP")
  add_ippl_integration_test(TestMixedExec LABELS integration field)
endif()
