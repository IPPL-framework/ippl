set (_SRCS
    )

set (_HDRS
    BareField.hpp
    BareField.h
    BareFieldOperations.hpp
    BConds.h
    BConds.hpp
    BcTypes.h
    BcTypes.hpp
    Field.h
    Field.hpp
    FieldOperations.hpp
    HaloCells.h
    HaloCells.hpp
    )

include_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/Field)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
