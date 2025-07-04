# -----------------------------------------------------------------------------
# src/Particle/CMakeLists.txt
#
# Adds particle container and layout headers to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install Particle-related headers
install(FILES
    ParticleAttribBase.h
    ParticleAttrib.h
    ParticleAttrib.hpp
    ParticleBase.h
    ParticleBase.hpp
    ParticleBC.h
    ParticleLayout.h
    ParticleLayout.hpp
    ParticleSpatialLayout.h
    ParticleSpatialLayout.hpp
    ParticleSpatialOverlapLayout.h
    ParticleSpatialOverlapLayout.hpp
    DESTINATION include/Particle
)

