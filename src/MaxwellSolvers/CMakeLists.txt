set (_SRCS
    )

set (_HDRS
    Maxwell.h
)

include_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/MaxwellSolvers)
