set (_SRCS
    # no source files
)

# file (GLOB_RECURSE _HDRS
#     "*.h"
#     "*.hpp"
# )

set (_HDRS
    # Elements
    Elements/Element.h
    Elements/HexahedralElement.h
    Elements/HexahedralElement.hpp
    Elements/EdgeElement.h
    Elements/EdgeElement.hpp
    Elements/QuadrilateralElement.h
    Elements/QuadrilateralElement.hpp

    # Quadrature
    Quadrature/GaussJacobiQuadrature.h
    Quadrature/GaussJacobiQuadrature.hpp
    Quadrature/MidpointQuadrature.h
    Quadrature/MidpointQuadrature.hpp
    Quadrature/Quadrature.h
    Quadrature/Quadrature.hpp

    # Spaces
    FiniteElementSpace.h
    FiniteElementSpace.hpp
    LagrangeSpace.h
    LagrangeSpace.hpp
    NedelecSpace.h
    NedelecSpace.hpp
    LagrangeSpaceFEMVector.h
    LagrangeSpaceFEMVector.hpp

    # Solvers
    FEMSolver.h
    FEMSolver.hpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/FEM)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
