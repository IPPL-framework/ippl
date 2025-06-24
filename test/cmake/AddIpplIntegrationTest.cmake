# -----------------------------------------------------------------------------
# AddIpplIntegrationTest.cmake
#
# Defines a helper function `add_ippl_integration_test()` to add integration tests.
# It builds an executable, links it with IPPL and MPI, includes headers from IPPL,
# and registers the test with CTest. Labels are optional and default to "integration".
# -----------------------------------------------------------------------------

function(add_ippl_integration_test TEST_NAME)
    set(options)
    set(oneValueArgs COMMAND LINK_DIRS)
    set(multiValueArgs LABELS ARGS)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)

    target_link_libraries(${TEST_NAME}
        PRIVATE ippl ${MPI_CXX_LIBRARIES}
    )

    if(TEST_LINK_DIRS)
        target_include_directories(${TEST_NAME}
            PRIVATE $<TARGET_PROPERTY:ippl,INTERFACE_INCLUDE_DIRECTORIES>
            ${TEST_LINK_DIRS}
        )
    else()
        target_include_directories(${TEST_NAME}
            PRIVATE $<TARGET_PROPERTY:ippl,INTERFACE_INCLUDE_DIRECTORIES>
        )
    endif()

    if(TEST_COMMAND)
        add_test(NAME Integration.${TEST_NAME} COMMAND ${TEST_COMMAND})
    else()
        add_test(NAME Integration.${TEST_NAME} COMMAND ${TEST_NAME} ${TEST_ARGS})
    endif()

    set_tests_properties(Integration.${TEST_NAME} PROPERTIES
        TIMEOUT 300
        LABELS "${TEST_LABELS}"
    )
endfunction()

