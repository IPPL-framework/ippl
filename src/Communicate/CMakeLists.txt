# -----------------------------------------------------------------------------
# src/Communicate/CMakeLists.txt
#
# Adds all Communicate-related sources and headers to the IPPL library.
# -----------------------------------------------------------------------------

target_sources(ippl PRIVATE Communicator.cpp CommunicatorLogging.cpp Environment.cpp Buffers.cpp
                            Request.cpp LogEntry.cpp)
