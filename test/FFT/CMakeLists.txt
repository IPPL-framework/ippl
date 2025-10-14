# -----------------------------------------------------------------------------
# test/FFT/CMakeLists.txt
#
# Integration tests for FFT functionality. These build example executables that run and validate FFT
# behavior across supported configurations.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding FFT integration tests from: ${_relPath}")

add_ippl_integration_test(TestFFTCC LABELS integration fft)
add_ippl_integration_test(TestFFTRC LABELS integration fft)
add_ippl_integration_test(TestSine LABELS integration fft)
add_ippl_integration_test(TestCos LABELS integration fft)
add_ippl_integration_test(TestCos1 LABELS integration fft)
