# -----------------------------------------------------------------------------
# SetupCatalyst.cmake
#
# Links Catalyst into the ippl target. Resolution (find vs FetchContent) happens
# in Dependencies.cmake.
# -----------------------------------------------------------------------------

if(NOT TARGET ippl)
  message(FATAL_ERROR "SetupCatalyst.cmake must be included after the ippl target is created.")
endif()

if(NOT TARGET catalyst::catalyst)
  message(
    FATAL_ERROR
      "Catalyst enabled but catalyst::catalyst target is missing. "
      "Set -DCATALYST_HINT_PATH=/path/to/catalyst/cmake or allow FetchContent.")
endif()

message(STATUS "Catalyst enabled")

target_compile_definitions(
  ippl
  PUBLIC IPPL_ENABLE_CATALYST
         CATALYST_ADAPTOR_ABS_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(ippl PUBLIC catalyst::catalyst)

message(STATUS "Catalyst Summary:")
message(STATUS "  External install: ${catalyst_FOUND}")
if(catalyst_VERSION)
  message(STATUS "  Version: ${catalyst_VERSION}")
elseif(Catalyst_VERSION)
  message(STATUS "  Version: ${Catalyst_VERSION}")
endif()
if(Catalyst_PACKAGE_DIR)
  message(STATUS "  CMake config dir: ${Catalyst_PACKAGE_DIR}")
endif()

get_target_property(_cat_type catalyst::catalyst TYPE)
message(STATUS "  Target: catalyst::catalyst (${_cat_type})")
unset(_cat_type)
