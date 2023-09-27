set (_SRCS
    # no source files at the moment
)

set (_HDRS
    Element.h
    Element.hpp
    FEMMesh.h
    FEMMEsh.hpp
    FEMSpace.h
    FEMSpace.hpp
    Hexahedron.h
    Hexahedron.hpp
    Quadrature.h
    Quadrature.hpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/FEM)
