file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(Field)
add_ippl_test(FieldBC)
add_ippl_test(Halo)
add_ippl_test(FieldSubLayout)
add_ippl_test(FieldSubLayoutBC)
add_ippl_test(HaloSubLayout)
