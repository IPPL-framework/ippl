message (STATUS "configure: make IpplInfo.h")
execute_process (
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND date
    OUTPUT_VARIABLE DATE_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
    )
execute_process (
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND uname -a
    OUTPUT_VARIABLE UNAME_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
    )
execute_process (
    COMMAND whoami
    OUTPUT_VARIABLE WHOAMI_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
    )

set (IPPL_COMPILE_ARCH \"$ENV{IPPL_ARCH}\")
set (IPPL_COMPILE_LINE \"${CMAKE_CXX_FLAGS}\")
set (IPPL_COMPILE_DATE \"${DATE_OUT}\")
set (IPPL_COMPILE_MACHINE \"${UNAME_OUT}\")
set (IPPL_COMPILE_OPTIONS \"${CMAKE_CXX_FLAGS}\")
set (IPPL_COMPILE_USER \"${WHOAMI_OUT}\")

configure_file (IpplVersions.h.in ${CMAKE_CURRENT_SOURCE_DIR}/IpplVersions.h)

set (IPPL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
macro (add_ippl_sources)
    file (RELATIVE_PATH _relPath "${IPPL_SRC_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND IPPL_SRCS "${_relPath}/${_src}")
        else ()
            list (APPEND IPPL_SRCS "${_src}")
        endif ()
    endforeach ()
    if (_relPath)
        # propagate SRCS to parent directory
        set (IPPL_SRCS ${IPPL_SRCS} PARENT_SCOPE)
    endif ()
endmacro ()

macro (add_ippl_headers)
    file (RELATIVE_PATH _relPath "${IPPL_SRC_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_hdr ${ARGN})
        if (_relPath)
            list (APPEND IPPL_HDRS "${_relPath}/${_hdr}")
        else ()
            list (APPEND IPPL_HDRS "${_hdr}")
        endif ()
    endforeach ()
    if (_relPath)
        # propagate HDRS to parent directory
        set (IPPL_HDRS ${IPPL_HDRS} PARENT_SCOPE)
    endif ()
endmacro ()

set (IPPL_BASEDIR_HDRS
    Ippl.h
    IpplVersions.h
    )
add_ippl_headers (${IPPL_BASEDIR_HDRS})
add_ippl_sources (Ippl.cpp)

add_subdirectory (Communicate)
if (ENABLE_FFT)
    set (HEFFTE_LIBRARY Heffte::Heffte)
    add_subdirectory (FFT)
else()
    set (HEFFTE_LIBRARY "")
endif()
add_subdirectory (Field)
add_subdirectory (FieldLayout)
add_subdirectory (Index)
add_subdirectory (Interpolation)
add_subdirectory (Meshes)
add_subdirectory (Particle)
add_subdirectory (Region)
add_subdirectory (Utility)
add_subdirectory (Expression)
add_subdirectory (Types)
add_subdirectory (Partition)
add_subdirectory (Random)

if (ENABLE_SOLVERS)
    message (STATUS "Adding Solver subdirectories")
    add_subdirectory (PoissonSolvers)
    add_subdirectory (MaxwellSolvers)
    add_subdirectory (LinearSolvers)
endif ()

if (ENABLE_AMR)
    add_subdirectory(AmrParticle)
endif ()

include_directories (
    BEFORE ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_library ( ippl ${IPPL_SRCS} ${IPPL_SRCS_FORT} )

set(TimerFences true)
string(REPLACE ";SERIAL" "" KokkosAccelerators "${Kokkos_DEVICES}")
string(REPLACE "SERIAL;" "" KokkosAccelerators "${KokkosAccelerators}")
if ("${KokkosAccelerators}" MATCHES ".+;.+")
    set(TimerFences false)
endif()
message("Timer fences for this build: ${TimerFences}")
target_compile_definitions(ippl PUBLIC IPPL_ENABLE_TIMER_FENCES=${TimerFences})
target_link_libraries(ippl PUBLIC Kokkos::kokkos ${HEFFTE_LIBRARY})

install (TARGETS ippl DESTINATION lib)
install (FILES ${IPPL_BASEDIR_HDRS} DESTINATION include)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode:cmake
# cmake-tab-width: 4
# indent-tabs-mode:nil
# require-final-newline: nil
# End:
