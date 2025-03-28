# -----------------------------------------------------------------------------
# CompilerOptions.cmake
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
#
# This file is only concerned with general correctness and development-time safety.
# -----------------------------------------------------------------------------

# === C++ Standard ===
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINGO} -O3 -g ")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set (CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# === Basic warnings (apply to all builds) ===
add_compile_options(
    -Wall
    -Wextra
    -Wno-deprecated-declarations
)

# === Compiler-specific warning suppressions ===
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -Wno-deprecated-copy
        -Wno-sign-compare
    )
endif()

# GCC 12+ false positives for buffer overflows, restrict, etc.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    add_compile_options(
        -Wno-stringop-overflow
        -Wno-array-bounds
        -Wno-restrict
    )
endif()

# === Debug-specific sanitizers ===
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "✅ Enabling AddressSanitizer and UBSan for Debug build")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

message(STATUS "✅ Compiler options configured")

