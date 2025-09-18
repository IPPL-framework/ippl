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
if(IPPL_ENABLE_COVERAGE AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  message(STATUS "${ColorYellow}Code coverage enabled.${ColorReset}")
  add_compile_options(-fprofile-arcs -ftest-coverage -g)
  add_link_options(-fprofile-arcs -ftest-coverage)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
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
