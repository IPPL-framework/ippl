# ------------------------------------------------------------------------------
# CTest custom settings
# ------------------------------------------------------------------------------

# Files/directories to exclude from CTest coverage reports. These regexes are applied to absolute
# source paths by ctest_coverage().
set(CTEST_CUSTOM_COVERAGE_EXCLUDE
    ".*/_deps/.*"
    ".*/unit_tests/.*"
    ".*/test/.*"
    ".*/demos/.*"
    ".*/alpine/.*"
    ".*/examples/.*"
    ".*/cosmology/.*"
    ".*/scripts/.*"
    ".*/CMakeFiles/.*")
