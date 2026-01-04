# -----------------------------------------------------------------------------
# src/CMakeLists.txt
# cmake-format: off
#
# This file builds the main IPPL library target. It:
#   - Generates IpplVersions.h from current build system info
#   - Gathers sources and headers
#   - Adds all component subdirectories (Communicate, Field, etc.)
#   - Links to external libraries (Kokkos, Heffte)
#   - Installs headers and the library
#
# cmake-format: on
# -----------------------------------------------------------------------------
message(STATUS "📄 Generating IpplVersions.h")

# Make embedded metadata optional (reduces rebuild churn).
option(IPPL_EMBED_BUILD_METADATA "Embed user/machine/date in IpplVersions.h" OFF)

if(IPPL_EMBED_BUILD_METADATA)

  execute_process(COMMAND date OUTPUT_VARIABLE IPPL_COMPILE_DATE OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND uname -a OUTPUT_VARIABLE IPPL_COMPILE_MACHINE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND whoami OUTPUT_VARIABLE IPPL_COMPILE_USER OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  set(IPPL_COMPILE_DATE "unknown")
  set(IPPL_COMPILE_MACHINE "unknown")
  set(IPPL_COMPILE_USER "unknown")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/IpplVersions.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/IpplVersions.h @ONLY)

add_library(ippl)

target_compile_features(ippl PUBLIC cxx_std_20)

target_compile_options(
  ippl
  PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
         $<$<CONFIG:Debug>:-O0
         -g>
         $<$<CONFIG:RelWithDebInfo>:-O3
         -g>
         $<$<CONFIG:Release>:-O3>
         >)

get_target_property(_ippl_type ippl TYPE)
if(_ippl_type)
  string(REPLACE "_LIBRARY" "" _ippl_type "${_ippl_type}") # -> STATIC / SHARED / OBJECT / INTERFACE
else()
  set(_ippl_type "UNKNOWN")
endif()

# Query PIC state (may inherit from CMAKE_POSITION_INDEPENDENT_CODE)
get_target_property(_ippl_pic ippl POSITION_INDEPENDENT_CODE)
if(NOT DEFINED _ippl_pic)
  set(_ippl_pic "DEFAULT(${CMAKE_POSITION_INDEPENDENT_CODE})")
endif()

message(STATUS "🔧 Target 'ippl': type=${_ippl_type}, POSITION_INDEPENDENT_CODE=${_ippl_pic}")

target_sources(ippl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Ippl.cpp)

target_include_directories(
  ippl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ippl>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(Communicate)

if(IPPL_ENABLE_FFT)
  add_subdirectory(FFT)
endif()

add_subdirectory(Field)
add_subdirectory(FieldLayout)
add_subdirectory(Decomposition)
add_subdirectory(Index)
add_subdirectory(Interpolation)
add_subdirectory(Manager)
add_subdirectory(Meshes)
add_subdirectory(Particle)
add_subdirectory(Region)
add_subdirectory(Utility)
add_subdirectory(Expression)
add_subdirectory(Types)
add_subdirectory(Partition)
add_subdirectory(FEM)
add_subdirectory(Random)

if(IPPL_ENABLE_SOLVERS)
  message(STATUS "Adding Solver subdirectories")
  add_subdirectory(PoissonSolvers)
  add_subdirectory(MaxwellSolvers)
  add_subdirectory(LinearSolvers)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/PlatformOptions.cmake)

target_link_libraries(ippl PUBLIC Kokkos::kokkos MPI::MPI_CXX)

if(IPPL_ENABLE_FFT)
  target_link_libraries(ippl PUBLIC Heffte::Heffte)
endif()

# this alias should be created after all target properties have been set
add_library(IPPL::ippl ALIAS ippl)

include(InstallIppl)
