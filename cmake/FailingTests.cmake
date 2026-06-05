# -----------------------------------------------------------------------------
# FailingTests.cmake
#
# Maintains a list of tests that are excluded from testing due to the fact that they are known to
# fail and spoil an otherwise green dashboard.
#
# Fixing these tests should be considered a priority
#
# -----------------------------------------------------------------------------

if(BUILD_TESTING AND IPPL_MARK_FAILING_TESTS)
  set(IPPL_DISABLED_TEST_LIST
  )

  set(IPPL_DISABLED_TEST_LIST_RELEASE ${IPPL_DISABLED_TEST_LIST})
  set(IPPL_DISABLED_TEST_LIST_DEBUG ${IPPL_DISABLED_TEST_LIST})
endif()
