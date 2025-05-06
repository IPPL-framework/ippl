# cosmology/test/CMakeLists.txt

file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "  - adding executable univ-3 found in ${_relPath}")


add_executable (univ-3  univ-3.cpp)
target_link_libraries (univ-3  ${IPPL_LIBS})
target_compile_options (univ-3 PRIVATE ${COMPILE_FLAGS})
