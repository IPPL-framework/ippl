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

# -----------------------------------------------------------------------------
# Symbolic version string (for CLI, logs, headers, etc.)
# -----------------------------------------------------------------------------
set(IPPL_VERSION_NAME "IPPL v${IPPL_VERSION}")

message(
  STATUS
    "📦 Configuring IPPL Version: ${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR} : \"${IPPL_VERSION_NAME}\""
)

# ------------------------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type (Debug, Release, RelWithDebInfo)")
  endif()
endif()

# ------------------------------------------------------------------------------
# C++ and language Standards
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# load message macros
# ------------------------------------------------------------------------------
include(Messages)

# ------------------------------------------------------------------------------
# Static/Dynamic build : use cmake's BUILD_SHARED_LIBS but set IPPL_LIB_TYPE for log purposes
# ------------------------------------------------------------------------------

# the user should use cmake's BUILD_SHARED_LIBS set(IPPL_LIB_TYPE STATIC)
# mark_as_advanced(IPPL_LIB_TYPE) if(BUILD_SHARED_LIBS) set(IPPL_LIB_TYPE SHARED) endif()
colour_message(STATUS ${Green} "🔧 IPPL will be built as a ${IPPL_LIB_TYPE} library")

# ------------------------------------------------------------------------------
# Default Build Type
# ------------------------------------------------------------------------------
set(_allowed_build_types Debug Release RelWithDebInfo MinSizeRel)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(default_build_type "RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "${default_build_type}"
      CACHE STRING "Choose the type of build (${_allowed_build_types})" FORCE)
  colour_message(STATUS ${Red} "No build type specified. Defaulting to ${CMAKE_BUILD_TYPE}")
else()
  colour_message(STATUS ${Green} "Build type is: ${CMAKE_BUILD_TYPE}")
endif()

if(NOT CMAKE_BUILD_TYPE IN_LIST _allowed_build_types)
  colour_message(WARNING ${Red} "Unknown CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# ------------------------------------------------------------------------------
message(STATUS "✅ Project setup complete${ColorReset}")
