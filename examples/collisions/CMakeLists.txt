file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding index test found in ${_relPath}")

add_executable(P3MHeating P3MHeating.cpp)

target_link_libraries(P3MHeating PRIVATE IPPL::ippl)
