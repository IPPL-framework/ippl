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

add_library(ippl::ippl ALIAS ippl)

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



message (STATUS "Adding Stream subdirectory")
add_subdirectory (Stream)


if (IPPL_ENABLE_CATALYST)
    message (STATUS "CATALYST enabled")
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
    message (STATUS "Found catalyst_DIR: ${catalyst_DIR}")
    if (catalyst_FOUND)
        message (STATUS "Found Catalyst:")
        target_compile_definitions(ippl PUBLIC
            IPPL_ENABLE_CATALYST 
        )

        message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
        target_compile_definitions(ippl PUBLIC
            CATALYST_ADAPTOR_ABS_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        )
        
        target_link_libraries(ippl PUBLIC 
            catalyst::catalyst
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
                    message(STATUS "  Include directories: ${_inc}")
                endif()

                # Config location
                if(catalyst_DIR)
                    message(STATUS "  CMake config dir: ${catalyst_DIR}")
                endif()

                # Target structure summary
                message(STATUS "  Target structure:")
                message(STATUS "    catalyst::catalyst (${_cat_type})")
                message(STATUS "    catalyst::catalyst_headers (INTERFACE_LIBRARY)")

    endif ()
endif()


if (IPPL_ENABLE_ASCENT)
    message (STATUS "ASCENT Enabled")


    # Create a list to hold all hint paths
    set(ASCENT_HINTS "")
    # If a path is provided from the command line, add it to the list first
    if(DEFINED ASCENT_HINT_PATH AND EXISTS "${ASCENT_HINT_PATH}")
        message(STATUS "Using user-provided Ascent hint path: ${ASCENT_HINT_PATH}")
        list(APPEND ASCENT_HINTS "${ASCENT_HINT_PATH}")
    endif()

    find_package(Ascent REQUIRED 
    HINTS ${ASCENT_HINTS}
    )
    # fails if ascent isn't found ...

    target_compile_definitions(ippl PUBLIC
        IPPL_ENABLE_ASCENT
    )
        # Use the correct target based on what's available
        if(TARGET ascent::ascent_mpi)
            target_link_libraries(ippl PUBLIC ascent::ascent_mpi)
            message(STATUS "Linking with ascent::ascent_mpi")
        elseif(TARGET ascent::ascent)
            target_link_libraries(ippl PUBLIC ascent::ascent)
            message(STATUS "Linking with ascent::ascent (no MPI)")
        else()
            message(FATAL_ERROR "No suitable Ascent target found")
        endif()
        ####################################################
        # Detailed INFO automatically printed by ascent ...#
        ####################################################

    
        ##-> done automatically ...
        ### INSTALL_PREFIX defined upon linking of the library...
    # include_directories ( 
    #     ${ASCENT_INSTALL_PREFIX}/include/ascent>
    # )


endif()



include(${PROJECT_SOURCE_DIR}/cmake/PlatformOptions.cmake)

target_link_libraries(ippl PUBLIC Kokkos::kokkos MPI::MPI_CXX)

if(IPPL_ENABLE_FFT)
  target_link_libraries(ippl PUBLIC Heffte::Heffte)
endif()

include(InstallIppl)
