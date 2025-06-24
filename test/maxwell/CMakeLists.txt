# -----------------------------------------------------------------------------
# test/maxwell/CMakeLists.txt
#
# Integration tests for IPPL maxwell solvers.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding maxwell integration tests from: ${_relPath}")

if(IPPL_ENABLE_FFT)
    add_ippl_integration_test(TestStandardFDTDSolver LINK_DIRS "${DOWNLOADED_HEADERS_DIR}" LABELS solver integration)
    add_ippl_integration_test(TestNonStandardFDTDSolver LINK_DIRS "${DOWNLOADED_HEADERS_DIR}" LABELS solver integration)
    add_ippl_integration_test(TestStandardFDTDSolver_convergence LINK_DIRS LABELS solver integration)
    add_ippl_integration_test(TestNonStandardFDTDSolver_convergence LINK_DIRS LABELS solver integration)
endif()