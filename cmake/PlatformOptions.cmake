# -----------------------------------------------------------------------------
# PlatformOptions.cmake
#
# Determines target-dependent compile-time flags (e.g., fencing behavior).
# -----------------------------------------------------------------------------

# Set default
set(TimerFences true)

# Determine from Kokkos device list if available
if(Kokkos_DEVICES)
  string(REPLACE ";SERIAL" "" KokkosAccelerators "${Kokkos_DEVICES}")
  string(REPLACE "SERIAL;" "" KokkosAccelerators "${KokkosAccelerators}")

  if("${KokkosAccelerators}" MATCHES ".+;.+")
    set(TimerFences false)
    message(
      STATUS "⏱️  Multiple Kokkos devices detected (${Kokkos_DEVICES}) — disabling timer fences.")
  else()
    message(STATUS "⏱️  Single Kokkos device detected (${Kokkos_DEVICES}) — enabling timer fences.")
  endif()
else()
  message(STATUS "⏱️  Kokkos_DEVICES not set — defaulting to TimerFences = true.")
endif()

# Define macro for use in source code
target_compile_definitions(ippl PUBLIC IPPL_ENABLE_TIMER_FENCES=${TimerFences})
