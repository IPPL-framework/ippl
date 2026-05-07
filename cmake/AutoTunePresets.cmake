# ============================================================================
# AutoTunePresets.cmake
#
# Detect the build's exec-space arch tag, copy any matching preset CSVs from
# `cmake/auto_tune/<tag>/` into the build's `share/ippl/auto_tune/`, and
# expose the resulting path to the library via the generated header
# `IpplAutoTunePresets.h`.
#
# At runtime, TileSizeCache::load() and GatherCache::load() consult that
# path after env / cwd lookups, so a fresh checkout on a known arch already
# uses tuned parameters without anyone running the sweep.
# ============================================================================

function(ippl_configure_autotune_presets)
  set(_ippl_cmake_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")

  # ---- Pick a tag that matches the layout under cmake/auto_tune/. -------
  #
  # Try Kokkos_ARCH_* cache variables first (they're reliable even when
  # CMAKE_CUDA_ARCHITECTURES / CMAKE_HIP_ARCHITECTURES is "native"). Fall
  # back to the numeric arch list if Kokkos didn't pin one.
  set(_tag "")

  if("CUDA" IN_LIST IPPL_PLATFORMS)
    set(_arch_map
        "KEPLER30:30"  "KEPLER32:32"  "KEPLER35:35"  "KEPLER37:37"
        "MAXWELL50:50" "MAXWELL52:52" "MAXWELL53:53"
        "PASCAL60:60"  "PASCAL61:61"
        "VOLTA70:70"   "VOLTA72:72"
        "TURING75:75"
        "AMPERE80:80"  "AMPERE86:86"  "AMPERE87:87"
        "ADA89:89"
        "HOPPER90:90"
        "BLACKWELL100:100" "BLACKWELL120:120")
    foreach(_entry ${_arch_map})
      string(REPLACE ":" ";" _pair "${_entry}")
      list(GET _pair 0 _name)
      list(GET _pair 1 _sm)
      if(Kokkos_ARCH_${_name})
        set(_tag "sm_${_sm}")
        break()
      endif()
    endforeach()

    if(NOT _tag AND CMAKE_CUDA_ARCHITECTURES)
      list(GET CMAKE_CUDA_ARCHITECTURES 0 _first_arch)
      string(REGEX REPLACE "[^0-9].*$" "" _first_arch "${_first_arch}")
      if(_first_arch)
        set(_tag "sm_${_first_arch}")
      endif()
    endif()
  elseif("HIP" IN_LIST IPPL_PLATFORMS)
    # Kokkos uses two naming conventions across versions: AMD_GFX* (newer)
    # and VEGA*/NAVI* (older). Check both.
    set(_hip_arch_map
        # AMD_GFX* (Kokkos >= ~4.x)
        "AMD_GFX906:gfx906"   "AMD_GFX908:gfx908"   "AMD_GFX90A:gfx90a"
        "AMD_GFX940:gfx940"   "AMD_GFX942:gfx942"
        "AMD_GFX1030:gfx1030" "AMD_GFX1100:gfx1100" "AMD_GFX1103:gfx1103"
        # VEGA*/NAVI* (older Kokkos)
        "VEGA906:gfx906"  "VEGA908:gfx908"  "VEGA90A:gfx90a"
        "VEGA940:gfx940"  "VEGA942:gfx942"
        "NAVI1030:gfx1030" "NAVI1100:gfx1100")
    foreach(_entry ${_hip_arch_map})
      string(REPLACE ":" ";" _pair "${_entry}")
      list(GET _pair 0 _name)
      list(GET _pair 1 _gfx)
      if(Kokkos_ARCH_${_name})
        set(_tag "${_gfx}")
        break()
      endif()
    endforeach()

    if(NOT _tag AND CMAKE_HIP_ARCHITECTURES)
      list(GET CMAKE_HIP_ARCHITECTURES 0 _first_arch)
      # CMAKE_HIP_ARCHITECTURES entries already look like "gfx90a"; strip
      # any trailing flags / colons just in case.
      string(REGEX REPLACE "[:].*$" "" _first_arch "${_first_arch}")
      if(_first_arch)
        set(_tag "${_first_arch}")
      endif()
    endif()
  elseif("OPENMP" IN_LIST IPPL_PLATFORMS)
    set(_tag "openmp")
  else()
    set(_tag "serial")
  endif()

  set(_src_dir "${_ippl_cmake_dir}/auto_tune/${_tag}")
  set(_dst_dir "${CMAKE_BINARY_DIR}/share/ippl/auto_tune")

  # Wipe any stale presets from a previous configure (e.g. arch changed or the
  # source preset directory was emptied). Otherwise the runtime would happily
  # keep loading a CSV produced for a different backend, leading to
  # team_size-too-large aborts on host backends.
  file(REMOVE
    "${_dst_dir}/tile_sweep_sa_optimal.csv"
    "${_dst_dir}/gather_sweep_optimal.csv")

  file(MAKE_DIRECTORY "${_dst_dir}")

  set(_have_scatter FALSE)
  set(_have_gather FALSE)

  if(_tag AND IS_DIRECTORY "${_src_dir}")
    if(EXISTS "${_src_dir}/tile_sweep_sa_optimal.csv")
      configure_file("${_src_dir}/tile_sweep_sa_optimal.csv"
                     "${_dst_dir}/tile_sweep_sa_optimal.csv" COPYONLY)
      set(_have_scatter TRUE)
    endif()
    if(EXISTS "${_src_dir}/gather_sweep_optimal.csv")
      configure_file("${_src_dir}/gather_sweep_optimal.csv"
                     "${_dst_dir}/gather_sweep_optimal.csv" COPYONLY)
      set(_have_gather TRUE)
    endif()
  endif()

  if(_have_scatter OR _have_gather)
    message(STATUS "📊 IPPL auto-tune presets: using ${_tag} ("
                   "scatter=${_have_scatter}, gather=${_have_gather})")
  else()
    if(_tag)
      message(STATUS "📊 IPPL auto-tune presets: none for ${_tag} (drop CSVs in ${_src_dir})")
    else()
      message(STATUS "📊 IPPL auto-tune presets: no tag resolved")
    endif()
  endif()

  # Bake into a generated header consumed by TileSizeCache / GatherCache.
  set(IPPL_AUTOTUNE_PRESET_DIR "${_dst_dir}")
  set(IPPL_AUTOTUNE_ARCH_TAG "${_tag}")
  configure_file(
    "${_ippl_cmake_dir}/IpplAutoTunePresets.h.in"
    "${CMAKE_BINARY_DIR}/include/IpplAutoTunePresets.h"
    @ONLY)

  # Install the preset directory next to the library so installed binaries
  # can find it (TileSizeCache also tries an install-relative fallback).
  install(DIRECTORY "${_dst_dir}/"
          DESTINATION "share/ippl/auto_tune"
          FILES_MATCHING PATTERN "*.csv")
endfunction()
