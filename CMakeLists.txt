cmake_minimum_required (VERSION 3.1)
project (IPPL CXX)
set (IPPL_VERSION_MAJOR 1)
set (IPPL_VERSION_MINOR 1.4)

set (IPPL_VERSION_NAME "V${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR}")

set (IPPL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE
        )
endif ()

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_definitions (-DNOPAssert)
endif ()

add_compile_options (-Wall)
add_compile_options (-Wunused)
add_compile_options (-Wextra)
add_compile_options (-Werror)

# allow deprecated functions
add_compile_options (-Wno-deprecated-declarations)

option (USE_STATIC_LIBRARIES "Link with static libraries if available" ON)

message (STATUS "Build type is: " ${CMAKE_BUILD_TYPE})

# Select flags.
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g ")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")


if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    set (CMAKE_CXX_STANDARD 20)
endif ()


# Resolve all library dependencies
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

find_package (MPI REQUIRED)
message (STATUS "The C++ compiler identification is: ${CMAKE_CXX_COMPILER_ID}")
message (STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
message (STATUS "The MPI C++ compiler is: ${MPI_CXX_COMPILER}")
message (STATUS "The underlying C++ compiler is: ${CMAKE_CXX_COMPILER}")


find_package(Kokkos REQUIRED)

option (ENABLE_FFT "Enable FFT transform" OFF)
if (ENABLE_FFT)
    add_definitions (-DENABLE_FFT)
    find_package(Heffte REQUIRED)
    message (STATUS "Found Heffte_DIR: ${Heffte_DIR}")
endif ()

set (BOOSTROOT $ENV{BOOST_DIR})
if (USE_STATIC_LIBRARIES)
    set (Boost_USE_STATIC_LIBS ON)
endif ()
set (Boost_USE_MULTITHREADED OFF)
set (Boost_USE_STATIC_RUNTIME OFF)
find_package (Boost 1.66.0
    REQUIRED COMPONENTS mpi filesystem iostreams regex serialization timer chrono system program_options)
message (STATUS "Found Boost include dir: ${Boost_INCLUDE_DIR}")
message (STATUS "Found Boost library dir: ${Boost_LIBRARY_DIRS}")
message (STATUS "Found Boost libraries: ${Boost_LIBRARIES}")
include_directories (SYSTEM ${Boost_INCLUDE_DIRS})

add_subdirectory (src)

option (ENABLE_IPPLTESTS "IPPL Tests" OFF)
if (ENABLE_IPPLTESTS)
    add_subdirectory (test)
endif ()

option (BUILD_UNIT_TESTS "Unit tests" OFF)
if (BUILD_UNIT_TESTS)
    find_package (GTest REQUIRED HINT $ENV{GTEST_PREFIX})
    add_subdirectory (unit_tests)
endif ()

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake )

install (
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}"
    RENAME ${PROJECT_NAME}Config.cmake
    )

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:

