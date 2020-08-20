set (_SRCS
    )

set (_HDRS
    ParticleAmrLayout.h
    AmrParticleBase.h
    AmrParticleBase.hpp
    AmrParticleLevelCounter.h
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/AmrParticle)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
