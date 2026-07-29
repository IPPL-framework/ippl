file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding examples found in ${_relPath}")

# Needed for threading
find_package(Threads REQUIRED)

# Macro for adding example executables
function(add_alpine_example name)
  add_executable(${name} ${name}.cpp)

  target_link_libraries(${name} PRIVATE IPPL::ippl)

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# === Add individual examples ===
add_alpine_example(UniformPlasmaTest)
add_alpine_example(LandauDampingMixedPrecision)
add_alpine_example(LandauDampingParameterList)

# LandauDampingMixedExec needs extra Threads lib
add_executable(LandauDampingMixedExec LandauDampingMixedExec.cpp)
target_link_libraries(LandauDampingMixedExec PRIVATE IPPL::ippl Threads::Threads)
target_include_directories(LandauDampingMixedExec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
