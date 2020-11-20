set (_SRCS
    Inform.cpp
    IpplInfo.cpp
#     IpplMemoryUsage.cpp
    IpplStats.cpp
    IpplTimings.cpp
    PAssert.cpp
    Timer.cpp
    Unique.cpp
    User.cpp
     UserList.cpp
    )

set (_HDRS
    Inform.h
    IpplException.h
    IpplInfo.h
#     IpplMemoryUsage.h
    IpplStats.h
    IpplTimings.h
#     NamedObj.h
    PAssert.h
    RefCounted.h
    Timer.h
    Unique.h
    User.h
    UserList.h
    my_auto_ptr.h
    vmap.h
    vmap.hpp
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
