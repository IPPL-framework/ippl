set (_SRCS
    DiscBuffer.cpp
    DiscConfig.cpp
    DiscMeta.cpp
    DiscParticleFunctions.cpp
    FieldDebugFunctions.cpp
    Inform.cpp
    IpplCounter.cpp
    IpplInfo.cpp
    IpplMemoryUsage.cpp
    IpplMessageCounter.cpp
    IpplStats.cpp
    IpplTimings.cpp
    PAssert.cpp
    ParticleDebugFunctions.cpp
    RNGXDiv.cpp
    RandomNumberGen.cpp
    StaticIpplInfo.cpp
    Timer.cpp
    Unique.cpp
    User.cpp
    UserList.cpp
    )

set (_HDRS
    DiscBuffer.h
    DiscConfig.h
    DiscField.h
    DiscField.hpp
    DiscMeta.h
    DiscParticle.h
    DiscType.h
    FieldDebug.h
    FieldDebug.hpp
    FieldDebugFunctions.h
    FieldDebugPrint.h
    FieldDebugPrint.hpp
    FieldPrint.h
    FieldPrint.hpp
    Inform.h
    IpplCounter.h
    IpplException.h
    IpplInfo.h
    IpplMemoryUsage.h
    IpplMessageCounter.h
    IpplStats.h
    IpplTimings.h
    NamedObj.h
    PAssert.h
    ParticleDebug.h
    ParticleDebug.hpp
    ParticleDebugFunctions.h
    RNGAssignDefs.h
    RNGXDiv.h
    RandomNumberGen.h
    RefCounted.h
    SequenceGen.h
    StaticIpplInfo.h
    Timer.h
    Unique.h
    User.h
    UserList.h
    Vec.h
    my_auto_ptr.h
    vmap.h
    vmap.hpp
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/Utility)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
