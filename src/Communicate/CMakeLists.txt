set (_SRCS
    Communicator.cpp
    CommunicatorLogging.cpp
    Environment.cpp
    Buffers.cpp
    Request.cpp
    LogEntry.cpp
    )

set (_HDRS
    LogEntry.h
    BufferHandler.h
    LoggingBufferHandler.h
    Archive.h
    Archive.hpp
    Buffers.hpp
    Communicator.h
    Environment.h
    DataTypes.h
    Operations.h
    Collectives.hpp
    Serializable.h
    Request.h
    Status.h
    TagMaker.h
    Tags.h
    Wait.h
    Window.h
    Window.hpp
    PointToPoint.hpp
    CommunicatorLogging.hpp
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
