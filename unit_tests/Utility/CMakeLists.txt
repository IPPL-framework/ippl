file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(ParameterList)
add_ippl_test(HeaderIncludeOrder COMPILE_ONLY)
