# -----------------------------------------------------------------------------
# src/Stream/InSitu/CMakeLists.txt
# -----------------------------------------------------------------------------
target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Add implementation sources for InSitu components
target_sources(ippl
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ProxyWriter.cpp
)


install(FILES
    ProxyWriter.h
    ProxyWriter.hpp
    CatalystAdaptor.h
    CatalystAdaptor.hpp
    CatalystVisitors.h
    AscentAdatpor.h
    VisBaseAdaptor.h
    DESTINATION include/Stream/InSitu 
)