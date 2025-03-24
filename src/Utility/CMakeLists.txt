# -----------------------------------------------------------------------------
# src/Utility/CMakeLists.txt
#
# Adds utility functions, assertions, timing, and info tools to the IPPL target.
# -----------------------------------------------------------------------------

# Add Utility source files to the ippl library
target_sources(ippl
    PRIVATE
        Inform.cpp
        IpplInfo.cpp
        IpplTimings.cpp
        PAssert.cpp
        Timer.cpp
)

target_include_directories(ippl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install Utility headers
install(FILES
    Inform.h
    IpplException.h
    IpplInfo.h
    IpplTimings.h
    PAssert.h
    Timer.h
    my_auto_ptr.h
    ParameterList.h
    TypeUtils.h
    ParallelDispatch.h
    ViewUtils.h
    DESTINATION include/Utility
)

