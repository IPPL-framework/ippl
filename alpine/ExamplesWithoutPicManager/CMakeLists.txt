file (RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

set(COMMON_LIBS
    ippl
    ${MPI_CXX_LIBRARIES}
)

set(COMMON_FLAGS
    ${OPAL_CXX_FLAGS}
)

# Needed for threading
find_package(Threads REQUIRED)

# Macro for adding example executables
function(add_alpine_example name)
    add_executable(${name} ${name}.cpp)

    target_compile_options(${name} PRIVATE ${COMMON_FLAGS})

    target_link_libraries(${name}
        PRIVATE
            ${COMMON_LIBS}
    )

    target_include_directories(${name}
        PRIVATE
            $<TARGET_PROPERTY:ippl,INTERFACE_INCLUDE_DIRECTORIES>
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endfunction()

# === Add individual examples ===
add_alpine_example(UniformPlasmaTest)
add_alpine_example(LandauDampingMixedPrecision)
add_alpine_example(LandauDampingParameterList)

# LandauDampingMixedExec needs extra Threads lib
add_executable(LandauDampingMixedExec LandauDampingMixedExec.cpp)
target_compile_options(LandauDampingMixedExec PRIVATE ${COMMON_FLAGS})
target_include_directories(LandauDampingMixedExec
    PRIVATE
        $<TARGET_PROPERTY:ippl,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(LandauDampingMixedExec
    PRIVATE
        ${COMMON_LIBS}
        Threads::Threads
)
