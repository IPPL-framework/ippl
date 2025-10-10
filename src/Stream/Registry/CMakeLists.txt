# -----------------------------------------------------------------------------
# src/Stream/InSitu/CMakeLists.txt
# -----------------------------------------------------------------------------
target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

install(FILES
    VisRegistryRuntime.h
    VisRegistry_mini.h
    VisRegistry.h
    VisRegistry.hpp
    ViewRegistry.h
    ViewRegistry.hpp
    RegistryHelper.h
    DESTINATION include/Stream/Registry
)