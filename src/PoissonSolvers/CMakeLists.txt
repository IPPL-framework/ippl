# -----------------------------------------------------------------------------
# src/PoissonSolvers/CMakeLists.txt
#
# Adds Poisson solver headers to the IPPL target, optionally with FFT backends.
# -----------------------------------------------------------------------------

# Nothing to do: public headers in this folder are visible via target_include_directories(ippl ...
# $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>) and are installed centrally by InstallIppl.cmake.
