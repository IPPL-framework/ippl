# -----------------------------------------------------------------------------
# Version.cmake
#
# Defines the version of the IPPL project in a centralized way.
#
# Why both IPPL_VERSION and PROJECT_VERSION? - PROJECT_VERSION is set by the top-level project(...)
# command and used by CMake. - IPPL_VERSION is our internal project version we can use freely in
# code, messages, version headers, etc., without relying on CMake internals.
# -----------------------------------------------------------------------------

set(IPPL_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(IPPL_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(IPPL_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(IPPL_VERSION ${PROJECT_VERSION})

# Symbolic version string (for CLI, logs, headers, etc.)
set(IPPL_VERSION_NAME "IPPL v${IPPL_VERSION}")

message(STATUS "📦 IPPL Version: ${IPPL_VERSION_NAME}")
