# -----------------------------------------------------------------------------
# demos/fel/CMakeLists.txt
#
# Free Electron Laser (FEL) demo application.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/fel/")

add_executable(FreeElectronLaser FreeElectronLaser.cpp)

target_link_libraries(FreeElectronLaser PRIVATE IPPL::ippl)

target_include_directories(FreeElectronLaser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/downloaded_headers)
