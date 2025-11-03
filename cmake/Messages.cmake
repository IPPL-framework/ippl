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
# cmake-format: off
if(NOT WIN32)
  string(ASCII 27 Esc)
  #
  set(ColourReset   "${Esc}[m")
  set(ColourBold    "${Esc}[1m")
  #
  set(Black         "${Esc}[30m")
  set(Red           "${Esc}[31m")
  set(Green         "${Esc}[32m")
  set(Yellow        "${Esc}[33m")
  set(Blue          "${Esc}[34m")
  set(Magenta       "${Esc}[35m")
  set(Cyan          "${Esc}[36m")
  set(White         "${Esc}[37m")
  #
  set(BoldRed       "${Esc}[1;31m")
  set(BoldGreen     "${Esc}[1;32m")
  set(BoldYellow    "${Esc}[1;33m")
  set(BoldBlue      "${Esc}[1;34m")
  set(BoldMagenta   "${Esc}[1;35m")
  set(BoldCyan      "${Esc}[1;36m")
  set(BoldWhite     "${Esc}[1;37m")
  #
  set(Gray          "${Esc}[90m")
  set(LightRed      "${Esc}[91m")
  set(LightGreen    "${Esc}[92m")
  set(LightYellow   "${Esc}[93m")
  set(LightBlue     "${Esc}[94m")
  set(LightMagenta  "${Esc}[95m")
  set(LightCyan     "${Esc}[96m")
endif()
# cmake-format: on

# ------------------------------------------------------------------------------
# Turn off colour when using ccmake
# ------------------------------------------------------------------------------
execute_process(COMMAND ps -o comm= OUTPUT_VARIABLE PS_OUT OUTPUT_STRIP_TRAILING_WHITESPACE)

if("${PS_OUT}" MATCHES "ccmake")
  set(_disable_colour ON)
endif()

# ------------------------------------------------------------------------------
# Coloured message
# ------------------------------------------------------------------------------
macro(colour_message TYPE COLOUR message)
  foreach(msg ${ARGN})
    set(msg_xtra "${msg_xtra} ${msg}")
  endforeach()
  if(DEFINED _disable_colour)
    message("${TYPE}" "${message}" "${msg_xtra}")
  else()
    message("${TYPE}" "${COLOUR}${message}${ColourReset}" "${msg_xtra}")
  endif()
endmacro()

# ------------------------------------------------------------------------------
# Debug message
# ------------------------------------------------------------------------------
function(debug_message DEBUG message)
  if(${DEBUG})
    colour_message(STATUS "${BoldGreen}" "DEBUG: ${message}")
  endif()
endfunction()
