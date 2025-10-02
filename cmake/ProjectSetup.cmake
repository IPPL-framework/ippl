# -----------------------------------------------------------------------------
# ProjectSetup.cmake
# cmake-format: off
#
# This file is responsible for foundational project settings that apply to all targets 
# and builds. It should be included near the top of the top-level CMakeLists.txt.
#
# Responsibilities:
#   - Set the C++ language standard and enforcement policy
#   - Set a default CMAKE_BUILD_TYPE if one is not specified
#   - Define macros for colored terminal output
#
# Not responsible for:
#   - Compiler warning/sanitizer flags    → CompilerOptions.cmake
#   - Platform/backend choices (e.g. CUDA)→ Platforms.cmake
#   - External dependencies               → Dependencies.cmake
#
# cmake-format: on
# -----------------------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type (Debug, Release, RelWithDebInfo)")
  endif()
endif()

# === C++ Standard ===
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(IPPL_LIB_TYPE STATIC)

option(USE_STATIC_LIBRARIES "Link with static libraries if available" ON)
option(Heffte_ENABLE_GPU_AWARE_MPI "Is a issue ... " OFF)

# === Default Build Type ===
set(_allowed_build_types Debug Release RelWithDebInfo MinSizeRel)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(default_build_type "RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "${default_build_type}"
      CACHE STRING "Choose the type of build (${_allowed_build_types})" FORCE)
  message(STATUS "No build type specified. Defaulting to ${CMAKE_BUILD_TYPE}")
else()
  message(STATUS "Build type is: ${CMAKE_BUILD_TYPE}")
endif()

if(NOT CMAKE_BUILD_TYPE IN_LIST _allowed_build_types)
  message(WARNING "Unknown CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# === Colored Output Macros ===
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(ColorRed "${Esc}[31m")
  set(ColorGreen "${Esc}[32m")
  set(ColorYellow "${Esc}[1;33m")
endif()

message(STATUS "${ColorGreen}✅ Project setup complete${ColorReset}")
