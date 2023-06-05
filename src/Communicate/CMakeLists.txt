set (_SRCS
    Communicate.cpp
    Buffers.cpp
    )

set (_HDRS
    Archive.h
    Archive.hpp
    Communicate.h
    DataTypes.h
    Operations.h
    Collectives.h
    Collectives.hpp
    TagMaker.h
    Tags.h
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/Communicate)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
