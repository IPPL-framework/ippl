cmake_minimum_required (VERSION 3.20.3)
project (IPPL CXX)
set (IPPL_VERSION_MAJOR 3)
set (IPPL_VERSION_MINOR 0.1)
include(FetchContent)
set (IPPL_VERSION_NAME "V${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR}")
option(DOWNLOAD_EXTRACT_TIMESTAMP ON FORCE)
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
#add_compile_options (-Werror)

# allow deprecated functions
add_compile_options (-Wno-deprecated-declarations)

option (USE_STATIC_LIBRARIES "Link with static libraries if available" ON)

message (STATUS "Build type is: " ${CMAKE_BUILD_TYPE})

if(IPPL_PLATFORMS)
    message(STATUS "IPPL_PLATFORMS flag is set to: ${IPPL_PLATFORMS}")
    string(TOUPPER "${IPPL_PLATFORMS}" IPPL_PLATFORMS)
    # Perform case distinction
    if("${IPPL_PLATFORMS}" STREQUAL "CUDA")
        message(STATUS "CUDA platform detected")
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA Backend")
        set(Heffte_ENABLE_CUDA ON CACHE BOOL "Set Heffte CUDA Backend")
    elseif("${IPPL_PLATFORMS}" STREQUAL "OPENMP")
        message(STATUS "OpenMP platform detected")
        if(NOT Heffte_ENABLE_FFTW)
            message(STATUS "Because Heffte_ENABLE_FFTW is not set defaulting to Stock with AVX")
            set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Enable Heffte Stock backend lol")
        endif()
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP Backend")
    elseif("${IPPL_PLATFORMS}" STREQUAL "SERIAL")
        message(STATUS "Serial platform detected")
        if(NOT Heffte_ENABLE_FFTW)
            message(STATUS "Because Heffte_ENABLE_FFTW is not set defaulting to Stock with AVX")
            set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Enable Heffte Stock backend lol")
        endif()
    elseif("${IPPL_PLATFORMS}" STREQUAL "CUDA;OPENMP" OR "${IPPL_PLATFORMS}" STREQUAL "OPENMP;CUDA")
        message(STATUS "CUDA and OpenMP platform detected")
        set(Heffte_ENABLE_CUDA ON CACHE BOOL "Set Heffte CUDA Backend")
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP Backend")
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA Backend") 
    else()
        message(WARNING "Unknown or unsupported platform: ${IPPL_PLATFORMS}")
        # Handle unknown or unsupported platform
    endif()
else()
    message(WARNING "IPPL_PLATFORMS flag is not set.")
    # Handle case where IPPL_PLATFORMS flag is not set
endif()

find_package(Kokkos 4.0.0 QUIET)
if(NOT Kokkos_FOUND)
    FetchContent_Declare(
        #DOWNLOAD_EXTRACT_TIMESTAMP
        kokko
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.1.00.tar.gz                
    )
    FetchContent_MakeAvailable(kokko)
endif()
# Select flags.
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINGO} -O3 -g ")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
#set (CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Use sanitizer flags for GCC or Clang only
    set (CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
endif ()

if ("${Kokkos_DEVICES}" MATCHES "CUDA")
    set (CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -G")
endif()

# Suppress erroneous buffer overflow warnings when compiling under GCC 12
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" MATCHES "12*")
    add_compile_options (-Wno-stringop-overflow)
    add_compile_options (-Wno-array-bounds)
    add_compile_options (-Wno-restrict)
endif()

# Resolve all library dependencies
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

find_package (MPI REQUIRED)
message (STATUS "The C++ compiler identification is: ${CMAKE_CXX_COMPILER_ID}")
message (STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
message (STATUS "The MPI C++ compiler is: ${MPI_CXX_COMPILER}")
message (STATUS "The underlying C++ compiler is: ${CMAKE_CXX_COMPILER}")

option(USE_ALTERNATIVE_VARIANT "Use modified variant implementation (required for CUDA 12.2 + GCC 12.3.0)" OFF)
if (USE_ALTERNATIVE_VARIANT)
    add_definitions (-DUSE_ALTERNATIVE_VARIANT)
endif()


option (ENABLE_FFT "Enable FFT transform" OFF)
if (ENABLE_FFT)
    add_definitions (-DENABLE_FFT)
    find_package (Heffte 2.2.0 QUIET)
    if(NOT Heffte_FOUND)
        message(STATUS "Using FetchContent for Heffte")
        FetchContent_Declare(
             Heffte
             #DOWNLOAD_EXTRACT_TIMESTAMP
             GIT_REPOSITORY https://github.com/icl-utk-edu/heffte 
        )
        FetchContent_MakeAvailable(Heffte)
    endif()
    message (STATUS "Found Heffte_DIR: ${Heffte_DIR}")
endif ()

option (ENABLE_SOLVERS "Enable IPPL solvers" OFF)

add_subdirectory (src)

option (ENABLE_TESTS "Enable IPPL tests" OFF)
if (ENABLE_TESTS)
    add_subdirectory (test)
endif ()

option (ENABLE_ALPINE "Enable Alpine" OFF)
if (ENABLE_ALPINE)
    if (NOT ENABLE_SOLVERS)
        message (FATAL_ERROR "Cannot enable Alpine since Solver not enabled (-DENABLE_SOLVERS=ON)!")
    endif ()
    if (NOT ENABLE_FFT)
        message (FATAL_ERROR "Cannot enable Alpine since FFT not enabled (-DENABLE_FFT=ON)!")
    endif ()
    message (STATUS "Enable Alpine")
    add_subdirectory (alpine)
endif ()

option (ENABLE_UNIT_TESTS "Enable unit tests" OFF)
if (ENABLE_UNIT_TESTS)
    include(FetchContent)
    include(GoogleTest)

    enable_testing()

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON # policy hint for CMake > 3.24
        )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

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

