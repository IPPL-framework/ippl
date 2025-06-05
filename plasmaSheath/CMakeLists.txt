# -----------------------------------------------------------------------------
# alpine/CMakeLists.txt
#
# This module provides example physics applications using IPPL.
# -----------------------------------------------------------------------------

set(COMMON_LIBS
    ippl
    ${MPI_CXX_LIBRARIES}
)

set(COMMON_FLAGS
    ${OPAL_CXX_FLAGS}
)

function(add_alpine_example name)
    add_executable(${name} ${name}.cpp)

    target_compile_options(${name} PRIVATE ${COMMON_FLAGS})

    target_link_libraries(${name}
        PRIVATE ${COMMON_LIBS}
    )

    target_include_directories(${name}
        PRIVATE
            $<TARGET_PROPERTY:ippl,INTERFACE_INCLUDE_DIRECTORIES>
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endfunction()

add_alpine_example(PlasmaSheath)
