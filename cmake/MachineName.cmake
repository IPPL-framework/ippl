# -----------------------------------------------------------------------------
# MachineName..cmake
# cmake-format: off
#
# Tries to find which machine we are running on so that scripts can be
# configured accordingly.
#
# Currently we support
# Lumi, alps (CSCS) machines,
#
# cmake-format: on
# -----------------------------------------------------------------------------

if(NOT "$ENV{CLUSTER_NAME}" STREQUAL "")
  set(MACHINE_NAME "$ENV{CLUSTER_NAME}")
elseif(NOT "$ENV{LMOD_SYSTEM_NAME}" STREQUAL "")
  set(MACHINE_NAME "$ENV{LMOD_SYSTEM_NAME}")
elseif(NOT "$ENV{APPS}" STREQUAL "")
  # basename of $APPS then append -xc
  get_filename_component(_apps_basename "$ENV{APPS}" NAME)
  set(MACHINE_NAME "${_apps_basename}-xc")
elseif(NOT "$ENV{LUMI_LMOD_FAMILY_COMPILER}" STREQUAL "")
  set(MACHINE_NAME "lumi")
else()
  execute_process(COMMAND hostname OUTPUT_VARIABLE _host OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(MACHINE_NAME "${_host}")
endif()

# export to environment and make available in cache for other CMake code
set(ENV{MACHINE_NAME} "${MACHINE_NAME}")
set(MACHINE_NAME "${MACHINE_NAME}" CACHE STRING "Detected machine name")

message(STATUS "MACHINE_NAME=${MACHINE_NAME}")
