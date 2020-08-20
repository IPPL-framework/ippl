set (_SRCS
    )

set (_HDRS
    BoxParticleCachingPolicy.h
    CellParticleCachingPolicy.h
    GenArrayParticle.h
    GenParticle.h
    IntCIC.h
    Interpolator.h
    IntNGP.h
    IntTSC.h
    IntSUDS.h
    NoParticleCachingPolicy.h
    PairBuilder/BasicPairBuilder.h
    PairBuilder/HashPairBuilder.h
    PairBuilder/HashPairBuilderPeriodic.h
    PairBuilder/PairConditions.h
    PairBuilder/SortingPairBuilder.h
    ParticleAttribBase.h
    ParticleAttrib.hpp
    ParticleAttribElem.h
    ParticleAttrib.h
    ParticleBalancer.hpp
    ParticleBalancer.h
    IpplParticleBase.hpp
    IpplParticleBase.h
    ParticleBConds.h
    ParticleCashedLayout.hpp
    ParticleCashedLayout.h
    ParticleInteractAttrib.hpp
    ParticleInteractAttrib.h
    ParticleInteractLayout.hpp
    ParticleInteractLayout.h
    ParticleLayout.hpp
    ParticleLayout.h
    ParticleSpatialLayout.hpp
    ParticleSpatialLayout.h
    ParticleUniformLayout.hpp
    ParticleUniformLayout.h
    PAssign.hpp
    PAssignDefs.h
    PAssign.h
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_ippl_sources (${_SRCS})
add_ippl_headers (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/Particle)

add_subdirectory (PairBuilder)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:

