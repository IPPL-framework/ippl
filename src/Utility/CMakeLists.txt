set (_SRCS
    Inform.cpp
    IpplInfo.cpp
    IpplTimings.cpp
    PAssert.cpp
    Timer.cpp
    )

set (_HDRS
    Inform.h
    IpplException.h
    IpplInfo.h
    IpplTimings.h
    PAssert.h
    Timer.h
    my_auto_ptr.h
    ParameterList.h
    TypeUtils.h
    ParallelDispatch.h
    ViewUtils.h
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/Utility)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
