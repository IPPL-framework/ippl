set (_SRCS
#     Centering.cpp
    )

set (_HDRS
    CartesianCentering.hpp
    CartesianCentering.h
    Centering.h
    Mesh.h
    Mesh.hpp
    UniformCartesian.h
    UniformCartesian.hpp
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/Meshes)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
