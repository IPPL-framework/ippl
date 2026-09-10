# -----------------------------------------------------------------------------
# demos/fel/CMakeLists.txt
#
# Free Electron Laser (FEL) demo application.
# -----------------------------------------------------------------------------
message(STATUS "Configuring demos/fel/")

add_executable(FreeElectronLaser FreeElectronLaser.cpp)

set(IPPL_FEL_CONFIG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/config.json")
set(IPPL_FEL_CONFIG_BUILD "${CMAKE_CURRENT_BINARY_DIR}/config.json")

add_custom_command(
  OUTPUT "${IPPL_FEL_CONFIG_BUILD}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${IPPL_FEL_CONFIG_SOURCE}"
          "${IPPL_FEL_CONFIG_BUILD}"
  DEPENDS "${IPPL_FEL_CONFIG_SOURCE}"
  COMMENT "Staging the FEL configuration"
  VERBATIM)

add_custom_target(ippl_fel_config DEPENDS "${IPPL_FEL_CONFIG_BUILD}")
add_dependencies(FreeElectronLaser ippl_fel_config)

target_link_libraries(
    FreeElectronLaser
    PRIVATE IPPL::ippl catalyst::catalyst)

target_include_directories(FreeElectronLaser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(FreeElectronLaser SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stb)
