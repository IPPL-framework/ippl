file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding unit tests found in ${_relPath}")

add_ippl_test(EdgeElement)
add_ippl_test(QuadrilateralElement)
add_ippl_test(HexahedralElement)
add_ippl_test(FiniteElementSpace)
add_ippl_test(LagrangeSpace)
add_ippl_test(Quadrature)
add_ippl_test(GaussJacobiQuadrature)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
