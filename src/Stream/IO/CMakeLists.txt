# -----------------------------------------------------------------------------
# src/Stream/IO/CMakeLists.txt
# -----------------------------------------------------------------------------
target_sources(ippl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ConduitIOWriter.cpp)

install(FILES ConduitIOWriter.h DESTINATION include/Stream/IO)
