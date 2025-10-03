file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(BufferHandler)
add_ippl_test(LoggingBufferHandler)
add_ippl_test(LogEntry)
