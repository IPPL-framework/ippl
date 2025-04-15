# -----------------------------------------------------------------------------
# AddIpplTest.cmake
#
# Defines a helper macro `add_ippl_test()` to create a unit test executable.
# It links to IPPL and GoogleTest, and sets include directories and labels.
# -----------------------------------------------------------------------------

function(add_ippl_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)

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

    gtest_discover_tests(${TEST_NAME}
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            TIMEOUT 600
            LABELS "${TEST_NAME};unit" 
    )
endfunction()

