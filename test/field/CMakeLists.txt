file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

if ("${Kokkos_DEVICES}" MATCHES "CUDA" AND "${Kokkos_DEVICES}" MATCHES "OPENMP")
    add_executable (TestMixedExec TestMixedExec.cpp)
    target_link_libraries (
        TestMixedExec
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )
endif()

add_executable (TestLaplace TestLaplace.cpp)
target_link_libraries (
    TestLaplace
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestFieldBC TestFieldBC.cpp)
target_link_libraries (
    TestFieldBC
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestHalo TestHalo.cpp)
target_link_libraries (
    TestHalo
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestCurl TestCurl.cpp)
target_link_libraries (
    TestCurl
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)

add_executable (TestHessian TestHessian.cpp)
target_link_libraries (
    TestHessian
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)
if(ENABLE_RENDERING)
    add_executable (TestRendering TestRendering.cpp)
    target_link_libraries (
        TestRendering
        ${IPPL_LIBS}
        ${MPI_CXX_LIBRARIES}
    )
endif()

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
