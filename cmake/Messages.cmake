# -----------------------------------------------------------------------------
# Messages.cmake
#
# This file is responsible for messages macros, color and debug.
#
# It should be included near the top of the top-level CMakeLists.txt.
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# OS specific settings for terminal colour
# ------------------------------------------------------------------------------
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Black "${Esc}[30m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(Light_Gray "${Esc}[37m")
  set(Gray "${Esc}[90m")
  set(Light_Red "${Esc}[91m")
  set(Light_Green "${Esc}[92m")
  set(Light_Yellow "${Esc}[93m")
  set(Light_Blue "${Esc}[94m")
  set(Light_Magenta "${Esc}[95m")
  set(Light_Cyan "${Esc}[96m")
  set(White "${Esc}[97m")
endif()

# ------------------------------------------------------------------------------
# Coloured message
# ------------------------------------------------------------------------------
macro(colour_message TYPE COLOUR message)
  message("${TYPE}" "${COLOUR}${message}${ColourReset}")
endmacro()

# ------------------------------------------------------------------------------
# Debug message
# ------------------------------------------------------------------------------
function(debug_message DEBUG message)
  if(${DEBUG})
    colour_message(STATUS "${Light_Green}" "DEBUG: ${message}")
  endif()
endfunction()
