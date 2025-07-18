# -----------------------------------------------------------------------------
# AddIpplTest.cmake
#
# Defines a helper macro `add_ippl_test()` to create a unit test executable.
# It links to IPPL and GoogleTest, and sets include directories and labels.
# -----------------------------------------------------------------------------

function(add_ippl_test TEST_NAME)
    set(options)
    set(oneValueArgs COMMAND)
    set(multiValueArgs LABELS ARGS)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)

    if(NOT TEST_NUM_PROCS)
        set(TEST_NUM_PROCS 2)
    endif()

    target_link_libraries(${TEST_NAME}
        PRIVATE
            ippl
            GTest::gtest
    )

    target_include_directories(${TEST_NAME}
        PRIVATE
            $<TARGET_PROPERTY:ippl,INTERFACE_INCLUDE_DIRECTORIES>
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    add_test(
      NAME ${TEST_NAME}
      COMMAND ${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${TEST_NUM_PROCS} "--allow-run-as-root" $<TARGET_FILE:${TEST_NAME}>
    )

    set(FINAL_LABELS unit ${TEST_LABELS})

    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 300
        LABELS "${FINAL_LABELS}"
    )


endfunction()

