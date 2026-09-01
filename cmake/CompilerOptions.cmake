# -----------------------------------------------------------------------------
# CompilerOptions.cmake
# cmake-format: off
#
# Sets compiler flags that affect how all IPPL targets are built.
#
# Responsibilities:
#   - Warning flags (-Wall, -Wextra, etc.)
#   - Debug sanitizers (ASan/UBSan) when using Debug build type
#   - Compiler-specific warning suppressions
#
# Not responsible for:
#   - Enabling CUDA/OpenMP/Serial                 → Platforms.cmake
#   - Selecting platform specific compiler flags  → Platforms.cmake 
#
# This file is only concerned with general correctness and development-time safety.
#
# cmake-format: on
# -----------------------------------------------------------------------------

# === Basic warnings (apply to all builds) ===
add_compile_options(-Wall -Wextra -Wno-deprecated-declarations)

# === Use modified variant implementation ===
if(IPPL_USE_ALTERNATIVE_VARIANT)
  add_definitions(-DIPPL_USE_ALTERNATIVE_VARIANT)
endif()

# === Code coverage options ===
# cmake-format: off
# Adds GCC/Clang gcov instrumentation to IPPL targets.
# -fprofile-arcs / -ftest-coverage : generate .gcno files at build time and .gcda files at runtime
# -fprofile-abs-path (GCC only)    : embed absolute source paths so gcov can still find the
#                                    sources when the build and test stages run in different dirs
# -g                               : keep line-number debug info
# The resulting data can be consumed by CTest/CDash (ctest_coverage) or by gcov/lcov manually.
# cmake-format: on
if(IPPL_ENABLE_COVERAGE)
  set(coverageSupported TRUE)

  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "IPPL_ENABLE_COVERAGE is only supported with GNU or Clang compilers.")
    set(coverageSupported FALSE)
  endif()

  if(IPPL_PLATFORMS MATCHES "(^|;)CUDA($|;)|(^|;)HIP($|;)")
    message(
      WARNING
        "IPPL_ENABLE_COVERAGE is requested with GPU platforms (${IPPL_PLATFORMS}). "
        "GPU device code is not instrumented by gcov; use a CPU-only build for meaningful coverage."
    )
    set(coverageSupported FALSE)
  endif()

  if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(
      WARNING
        "IPPL_ENABLE_COVERAGE is enabled with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "Compiler optimization may produce misleading coverage; use Debug for accurate results.")
  endif()

  if(coverageSupported)
    message(STATUS "${ColorYellow}Code coverage enabled.${ColorReset}")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      add_compile_options(-fprofile-abs-path)
    endif()
    add_compile_options(-fprofile-arcs -ftest-coverage -g)
    add_link_options(-fprofile-arcs -ftest-coverage)
  endif()
endif()

# === Compiler-specific warning suppressions ===
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wno-deprecated-copy -Wno-sign-compare)
endif()

# GCC 12+ false positives for buffer overflows, restrict, etc.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
  add_compile_options(-Wno-stringop-overflow -Wno-array-bounds -Wno-restrict)
endif()

# === Debug-specific sanitizers ===
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU"
   AND IPPL_ENABLE_SANITIZER)
  message(STATUS "✅ Enabling AddressSanitizer and UBSan for Debug build")
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

# === Position Independent Code (PIC) for shared libraries ===
if(BUILD_SHARED_LIBS)
  message(STATUS "✅ Enabling Position Independent Code (PIC) for shared libraries")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_LINK_DEPENDS_NO_SHARED true)
endif()

message(STATUS "✅ Compiler options configured")
