# -----------------------------------------------------------------------------
# src/Communicate/CMakeLists.txt
#
# Adds all Communicate-related sources and headers to the IPPL library.
# -----------------------------------------------------------------------------

target_sources(ippl
    PRIVATE
        Communicator.cpp
        CommunicatorLogging.cpp
        Environment.cpp
        Buffers.cpp
        Request.cpp
        LogEntry.cpp
)

target_link_libraries(ippl PRIVATE MPI::MPI_CXX)

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install public headers
install(FILES
    LogEntry.h
    BufferHandler.h
    BufferHandler.hpp
    LoggingBufferHandler.h
    LoggingBufferHandler.hpp
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
    DESTINATION include/Communicate
)

