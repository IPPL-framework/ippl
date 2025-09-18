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

if (IPPL_ENABLE_CATALYST)
    message (STATUS "Adding Stream subdirectory")
    add_subdirectory (Stream)


    # Create a list to hold all hint paths
    set(CATALYST_HINTS "")

    # If a path is provided from the command line, add it to the list first
    if(DEFINED CATALYST_HINT_PATH AND EXISTS "${CATALYST_HINT_PATH}")
        message(STATUS "Using user-provided Catalyst hint path: ${CATALYST_HINT_PATH}")
        list(APPEND CATALYST_HINTS "${CATALYST_HINT_PATH}")
    endif()


    # Use the list of hints in find_package
    find_package(catalyst REQUIRED
        HINTS 
        ${CATALYST_HINTS}
    )

    if (catalyst_FOUND)
        message (STATUS "Found Catalyst:")
        target_link_libraries(ippl PUBLIC 
            catalyst::catalyst
        )
        target_compile_definitions(ippl PUBLIC
            IPPL_ENABLE_CATALYST 
        )


                # ======================================================
                # ========= CATALYST INFO SUMMARY ======================
                # ======================================================

                message(STATUS "Catalyst Summary:")
                message(STATUS "  Found: ${catalyst_FOUND}")
                message(STATUS "  Version: ${catalyst_VERSION}")

                # Display debug library location
                get_property(_cfgs TARGET catalyst::catalyst PROPERTY IMPORTED_CONFIGURATIONS)
                foreach(_cfg IN LISTS _cfgs)
                    get_target_property(_loc catalyst::catalyst IMPORTED_LOCATION_${_cfg})
                    if(_loc)
                        message(STATUS "  Library (${_cfg}): ${_loc}")
                    endif()
                endforeach()

                # Headers information
                message(STATUS "  Headers:")
                get_target_property(_inc catalyst::catalyst_headers INTERFACE_INCLUDE_DIRECTORIES)
                if(_inc)
                    message(STATUS "    Include directories: ${_inc}")
                endif()

                # Config location
                if(catalyst_DIR)
                    message(STATUS "  CMake config dir: ${catalyst_DIR}")
                endif()

                # Target structure summary
                message(STATUS "  Target structure:")
                message(STATUS "    catalyst::catalyst (${_cat_type})")
                message(STATUS "    catalyst::catalyst_headers (INTERFACE_LIBRARY)")

                # ======================================================


    else()
        # Fallback: locate headers and library manually
        message(STATUS "WARNING: CATALYST NOT FOUND, find_package failed")
    endif ()
endif()


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
