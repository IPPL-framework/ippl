# -----------------------------------------------------------------------------
# demos/fel/CMakeLists.txt
#
# Free Electron Laser (FEL) demo application.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/fel/")

add_executable(FreeElectronLaser FreeElectronLaser.cpp)

target_link_libraries(
    FreeElectronLaser
    PRIVATE IPPL::ippl nlohmann_json::nlohmann_json)

target_include_directories(FreeElectronLaser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(FreeElectronLaser SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stb)
