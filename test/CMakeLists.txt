# -----------------------------------------------------------------------------
# test/CMakeLists.txt
#
# These are NOT unit tests. These are full simulation examples or integration
# tests that check combined module functionality. They build standalone
# executables and are run using `ctest` to check successful execution.
# -----------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/test/cmake")
include(AddIpplIntegrationTest)

add_subdirectory(kokkos)
add_subdirectory(types)
add_subdirectory(field)

if (IPPL_ENABLE_FFT)
    add_subdirectory(FFT)
endif()

if (IPPL_ENABLE_SOLVERS)
    add_subdirectory(solver)
    add_subdirectory(solver/fem)
endif()

add_subdirectory(particle)
add_subdirectory(region)
add_subdirectory(random)
add_subdirectory(serialization)
add_subdirectory(FEM)

