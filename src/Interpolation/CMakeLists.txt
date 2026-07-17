# -----------------------------------------------------------------------------
# src/Interpolation/CMakeLists.txt
#
# Adds particle-to-grid interpolation headers to the IPPL target.
# -----------------------------------------------------------------------------

target_sources(ippl PRIVATE Scatter/AutoTune.cpp Scatter/TileSizeCache.cpp)

# Public headers in this folder are visible via target_include_directories(ippl ...
# $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>) and are installed centrally by InstallIppl.cmake.
