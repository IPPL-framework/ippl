cmake_minimum_required (VERSION 3.20.3)
project (IPPL CXX)
set (IPPL_VERSION_MAJOR 3)
set (IPPL_VERSION_MINOR 0.2)
include(FetchContent)
set (IPPL_VERSION_NAME "V${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR}")
option(DOWNLOAD_EXTRACT_TIMESTAMP ON FORCE)
set (IPPL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[1;33m")
endif()
macro(remove_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    if(_target_cxx_flags)
        list(REMOVE_ITEM _target_cxx_flags ${_flag})
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
endmacro()


if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

    set (CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE
    )
    message(STATUS "${Yellow}CMAKE_BUILD_TYPE not set, defaulting to ${CMAKE_BUILD_TYPE} ${ColourReset}")
else ()
    message (STATUS "${Green}Build type is: ${CMAKE_BUILD_TYPE}${ColourReset}")
endif()


if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_definitions (-DNOPAssert)
endif ()

add_compile_options (-Wall)
add_compile_options (-Wunused)
add_compile_options (-Wextra)
#¼add_compile_options (-Werror)

# allow deprecated functions
add_compile_options (-Wno-deprecated-declarations)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options (-Wno-deprecated-copy)
    add_compile_options (-Wno-sign-compare)
endif()

option (USE_STATIC_LIBRARIES "Link with static libraries if available" ON)

if(NOT IPPL_PLATFORMS)
    message(STATUS "${Yellow}IPPL_PLATFORMS flag is not set, defaulting to serial backend${ColourReset}")
    set(IPPL_PLATFORMS "SERIAL")
endif()


#message(STATUS "IPPL_PLATFORMS flag is set to: ${IPPL_PLATFORMS}")
if(Ippl_PLATFORMS)
    set(IPPL_PLATFORMS ${Ippl_PLATFORMS})
endif()
string(TOUPPER "${IPPL_PLATFORMS}" IPPL_PLATFORMS)
# Perform case distinction
if("${IPPL_PLATFORMS}" STREQUAL "CUDA")
    message(STATUS "${Green}CUDA backend enabled${ColourReset}")
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA Backend")
    set(Heffte_ENABLE_CUDA ON CACHE BOOL "Set Heffte CUDA Backend")

elseif("${IPPL_PLATFORMS}" STREQUAL "OPENMP")
    message(STATUS "${Green}OpenMP backend enabled${ColourReset}")
    if(NOT Heffte_ENABLE_FFTW)
        message(STATUS "${Yellow}Because Heffte_ENABLE_FFTW is not set defaulting to Stock with AVX${ColourReset}")
        set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Enable Heffte Stock backend lol")
    else()
        message(STATUS "${Green}Enabled FFTW${ColourReset}")
    endif()
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP Backend")

elseif("${IPPL_PLATFORMS}" STREQUAL "SERIAL")
    message(STATUS "${Green}Serial backend enabled${ColourReset}")
    if(NOT Heffte_ENABLE_FFTW)
        message(STATUS "${Yellow}Because Heffte_ENABLE_FFTW is not set defaulting to Stock with AVX${ColourReset}")
        set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Enable Heffte Stock backend lol")
    endif()

elseif("${IPPL_PLATFORMS}" STREQUAL "CUDA;OPENMP" OR "${IPPL_PLATFORMS}" STREQUAL "OPENMP;CUDA")
    message(STATUS "${Green}CUDA and OpenMP backend enabled${ColourReset}")
    set(Heffte_ENABLE_CUDA ON CACHE BOOL "Set Heffte CUDA Backend")
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP Backend")
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA Backend")

else()
    message(FATAL_ERROR "Unknown or unsupported platform: ${IPPL_PLATFORMS}")
    # Handle unknown or unsupported platform
endif()

if(NOT Kokkos_VERSION)
    set(Kokkos_VERSION "4.1.00")
    message(STATUS "${Green}Selecting default Kokkos version: ${Kokkos_VERSION} ${ColourReset}")
else()
    message(STATUS "${Green}Selected Kokkos version: ${Kokkos_VERSION} ${ColourReset}")
endif()
find_package(Kokkos ${Kokkos_VERSION} QUIET)
if(NOT Kokkos_FOUND)
    FetchContent_Declare(
        kokkos
        DOWNLOAD_EXTRACT_TIMESTAMP True
        URL https://github.com/kokkos/kokkos/archive/refs/tags/${Kokkos_VERSION}.tar.gz                
    )
    FetchContent_MakeAvailable(kokkos)
endif()
# Select flags.
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_EXTENSIONS OFF)
set (CMAKE_CXX_EXTENSIONS OFF)
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
    if(NOT Heffte_VERSION)
        set(Heffte_VERSION "MASTER")
        message(STATUS "${Green}Selecting default Heffte version: MASTER (Master branch on ttps://github.com/icl-utk-edu/heffte) ${ColourReset}")
    else()
    message(STATUS "${Green}Selected Heffte version: ${Heffte_VERSION} ${ColourReset}")
    endif()
    if(Heffte_VERSION MATCHES "MASTER")
    find_package (Heffte 2.2.0 QUIET)
    else()
    find_package (Heffte ${Heffte_VERSION} QUIET)
    endif()
    if(NOT Heffte_FOUND)
        message(STATUS "Using FetchContent for Heffte")
        if(Heffte_VERSION MATCHES "MASTER")
        if(NOT HEFFTE_COMMIT_HASH)
            set(HEFFTE_COMMIT_HASH "9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda")
            message(STATUS "${Yellow}Defaulting to Heffte commit: ${HEFFTE_COMMIT_HASH} ${ColourReset}")
        else()
            message(STATUS "${Green}Using Heffte commit: ${HEFFTE_COMMIT_HASH} ${ColourReset}")
        endif()
        FetchContent_Declare(
            Heffte
            GIT_REPOSITORY https://github.com/icl-utk-edu/heffte
            GIT_TAG ${HEFFTE_COMMIT_HASH}
        )
        else()
        FetchContent_Declare(
            Heffte
            DOWNLOAD_EXTRACT_TIMESTAMP True
            URL https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${Heffte_VERSION}.tar.gz
        )
        endif()
        FetchContent_MakeAvailable(Heffte)
        # Required to suppress warnings related to heffte
        target_include_directories(Heffte SYSTEM PUBLIC "${Heffte_SOURCE_DIR}/include")
        set(Heffte_DIR "${Heffte_SOURCE_DIR}")
        remove_flag_from_target(Heffte "-Werror")
        remove_flag_from_target(Heffte "-Wall")
        remove_flag_from_target(Heffte "-Wextra")
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
    if (NOT ENABLE_FFT)
        message(WARNING "Enabling unit tests even though FFT not enabled (-DENABLE_FFT=ON)!")
    endif ()
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

