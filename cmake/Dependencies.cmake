# -----------------------------------------------------------------------------
# Dependencies.cmake
#
# Resolves third-party libraries: Kokkos and Heffte.
#
# Responsibilities:
#   - Fetch or find Kokkos, using version and backends from Platforms.cmake
#   - Fetch Heffte if IPPL_ENABLE_FFT is ON, using CUDA or AVX2 based on platform
#
# Not responsible for:
#   - Selecting platform backends            → Platforms.cmake
#   - Enabling compiler flags                → CompilerOptions.cmake
#   - Version variables or target creation   → Version.cmake / src/
# -----------------------------------------------------------------------------
include(FetchContent)

# === Kokkos ===
set(Kokkos_VERSION "4.5.00")
message(STATUS "🔍 Looking for Kokkos ${Kokkos_VERSION}")

find_package(Kokkos ${Kokkos_VERSION} QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "📥 Kokkos not found — using FetchContent")
    FetchContent_Declare(
        kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/${Kokkos_VERSION}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    FetchContent_MakeAvailable(kokkos)
endif()

message(STATUS "✅ Kokkos ready")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Heffte_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
    message(STATUS "❗ Disabling AVX2 and CUDA in Debug build")
endif()

# === Heffte (only if FFT enabled) ===
if(IPPL_ENABLE_FFT)
    add_compile_definitions(IPPL_ENABLE_FFT)

    if(NOT DEFINED Heffte_VERSION)
        set(Heffte_VERSION "master")
        set(HEFFTE_COMMIT_HASH "9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda")
        set(USE_FETCH_HEFFTE TRUE)
        message(STATUS "📦 Using Heffte master from commit ${HEFFTE_COMMIT_HASH}")
    else()
        find_package(Heffte ${Heffte_VERSION} CONFIG QUIET)
        if(Heffte_FOUND)
            message(STATUS "✅ Found installed Heffte ${Heffte_VERSION}")
        else()
            message(STATUS "📦 Heffte ${Heffte_VERSION} not found, fetching...")
            set(USE_FETCH_HEFFTE TRUE)
        endif()
    endif()

    if(USE_FETCH_HEFFTE)
        include(FetchContent)

        # Define backend options BEFORE calling MakeAvailable
        if(NOT DEFINED Heffte_ENABLE_FFTW AND NOT DEFINED Heffte_ENABLE_CUDA AND NOT DEFINED Heffte_ENABLE_MKL)
            set(Heffte_ENABLE_AVX2 ON CACHE BOOL "Use AVX2 backend in Heffte" FORCE)
            set(Heffte_ENABLE_FFTW OFF CACHE BOOL "Use FFTW in Heffte" FORCE)
            set(Heffte_ENABLE_CUDA OFF CACHE BOOL "Use CUDA in Heffte" FORCE)
        endif()

        if(Heffte_VERSION STREQUAL "master")
            FetchContent_Declare(
                heffte
                GIT_REPOSITORY https://github.com/icl-utk-edu/heffte.git
                GIT_TAG ${HEFFTE_COMMIT_HASH}
                DOWNLOAD_EXTRACT_TIMESTAMP ON
            )
        else()
            FetchContent_Declare(
                heffte
                URL https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${Heffte_VERSION}.tar.gz
                DOWNLOAD_EXTRACT_TIMESTAMP ON
            )
        endif()

        FetchContent_MakeAvailable(heffte)
        if(TARGET Heffte AND NOT TARGET Heffte::heffte)
            add_library(Heffte::heffte ALIAS Heffte)
            message(STATUS "🔗 Created ALIAS Heffte::heffte for Heffte target.")
        endif()

        # Safety check
        if(NOT TARGET Heffte::heffte)
            message(FATAL_ERROR "❌ Heffte::heffte target is missing. Check Heffte build configuration.")
        endif()

        message(STATUS "✅ Heffte built from source (${Heffte_VERSION})")
    endif()

endif()


if(IPPL_ENABLE_UNIT_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    message(STATUS "✅ GoogleTest loaded for unit tests.")
endif()

if(IPPL_ENABLE_TESTS)
    set(DOWNLOADED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/downloaded_headers")
    file(DOWNLOAD
        https://raw.githubusercontent.com/manuel5975p/stb/master/stb_image_write.h
        "${DOWNLOADED_HEADERS_DIR}/stb_image_write.h"
    )
    message(STATUS "✅ stb_image_write loaded for testing FDTD solver.")
endif()
