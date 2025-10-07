# -----------------------------------------------------------------------------
# src/Stream/InSitu/CMakeLists.txt
# -----------------------------------------------------------------------------
target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)


install(FILES
    CatalystAdaptor.h
    CatalystAdaptor.hpp
    AscentAdatpor.h
    VisBaseAdaptor.h
    DESTINATION include/Stream/InSitu 
)