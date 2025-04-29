# -----------------------------------------------------------------------------
# src/CMakeLists.txt
#
# This file builds the main IPPL library target. It:
#   - Generates IpplVersions.h from current build system info
#   - Gathers sources and headers
#   - Adds all component subdirectories (Communicate, Field, etc.)
#   - Links to external libraries (Kokkos, Heffte)
#   - Installs headers and the library
# -----------------------------------------------------------------------------
message(STATUS "📄 Generating IpplVersions.h")

execute_process(
    COMMAND date
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE IPPL_COMPILE_DATE OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND uname -a
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE IPPL_COMPILE_MACHINE OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND whoami
    OUTPUT_VARIABLE IPPL_COMPILE_USER OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(IPPL_COMPILE_ARCH "$ENV{IPPL_ARCH}")
set(IPPL_COMPILE_LINE "${CMAKE_CXX_FLAGS}")
set(IPPL_COMPILE_OPTIONS "${CMAKE_CXX_FLAGS}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/IpplVersions.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/IpplVersions.h
)

add_library(ippl ${IPPL_LIB_TYPE})

target_sources(ippl
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Ippl.cpp
)

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
)

add_subdirectory (Communicate)

if (IPPL_ENABLE_FFT)
    add_subdirectory (FFT)
endif()

add_subdirectory (Field)
add_subdirectory (FieldLayout)
add_subdirectory (Decomposition)
add_subdirectory (Index)
add_subdirectory (Interpolation)
add_subdirectory (Manager)
add_subdirectory (Meshes)
add_subdirectory (Particle)
add_subdirectory (Region)
add_subdirectory (Utility)
add_subdirectory (Expression)
add_subdirectory (Types)
add_subdirectory (Partition)
add_subdirectory (FEM)
add_subdirectory (Random)

if (IPPL_ENABLE_SOLVERS)
    message (STATUS "Adding Solver subdirectories")
    add_subdirectory (PoissonSolvers)
    add_subdirectory (MaxwellSolvers)
    add_subdirectory (LinearSolvers)
endif ()

include(${PROJECT_SOURCE_DIR}/cmake/PlatformOptions.cmake)

target_link_libraries(ippl
    PUBLIC
        Kokkos::kokkos
)

if(IPPL_ENABLE_FFT AND TARGET Heffte::heffte)
    target_link_libraries(ippl PUBLIC Heffte::heffte)
else()
    message(WARNING "⚠️ Heffte::heffte target not found — skipping link.")
endif()

set(IPPL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(IPPL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${PROJECT_SOURCE_DIR}/cmake/InstallIppl.cmake)
