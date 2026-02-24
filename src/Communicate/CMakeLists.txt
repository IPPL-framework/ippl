# -----------------------------------------------------------------------------
# src/Communicate/CMakeLists.txt
#
# Adds all Communicate-related sources and headers to the IPPL library.
# -----------------------------------------------------------------------------

if(IPPL_ENABLE_ALIGNED_COMMUNICATION_BUFFERS)
  target_compile_definitions(ippl PUBLIC IPPL_ALIGNED_COMMS_BUFFERS)
endif()

target_sources(ippl PRIVATE Communicator.cpp CommunicatorLogging.cpp Environment.cpp Buffers.cpp
                            Request.cpp LogEntry.cpp)
