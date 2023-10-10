set (_SRCS
    # no source files
)

set (GLOB_RECURSE _HDRS
    "*.h"
    "*.hpp"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/FEM)
