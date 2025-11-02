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
  set(IPPL_MACHINENAME "$ENV{CLUSTER_NAME}")
elseif(NOT "$ENV{LMOD_SYSTEM_NAME}" STREQUAL "")
  set(IPPL_MACHINENAME "$ENV{LMOD_SYSTEM_NAME}")
elseif(NOT "$ENV{APPS}" STREQUAL "")
  # basename of $APPS then append -xc
  get_filename_component(_apps_basename "$ENV{APPS}" NAME)
  set(IPPL_MACHINENAME "${_apps_basename}-xc")
elseif(NOT "$ENV{LUMI_LMOD_FAMILY_COMPILER}" STREQUAL "")
  set(IPPL_MACHINENAME "lumi")
else()
  execute_process(COMMAND hostname OUTPUT_VARIABLE _host OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(IPPL_MACHINENAME "${_host}")
endif()

# export to environment and make available in cache for other CMake code
set(ENV{IPPL_MACHINENAME} "${IPPL_MACHINENAME}")
set(IPPL_MACHINENAME "${IPPL_MACHINENAME}" CACHE STRING "Detected machine name")
