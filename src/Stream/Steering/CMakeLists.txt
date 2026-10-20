target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

if(IPPL_ENABLE_CATALYST)
    # ProxyWriter uses Catalyst's Conduit API.
    target_sources(ippl
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/ProxyWriter.cpp
    )
endif()

install(FILES
    Button.h
    ProxyWriter.h
    ProxyWriter.hpp
    DESTINATION include/Stream/Steering
)
