set (_SRCS
  VRB.cpp
  )

set (_HDRS
  BinaryBalancer.hpp
  BinaryBalancer.h
  CenteredFieldLayout.hpp
  CenteredFieldLayout.h
  FieldLayout.hpp
  FieldLayout.h
  FieldLayoutUser.h
  Vnode.h
  VRB.h
  )

include_DIRECTORIES (
  ${CMAKE_CURRENT_SOURCE_DIR}
)

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/FieldLayout)

# vi: set et ts=4 sw=4 sts=4:
  
# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
