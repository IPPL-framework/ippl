# -----------------------------------------------------------------------------
# src/Interaction/CMakeLists.txt
#
# Adds ParticleParticle Interactions to the IPPL target.
# -----------------------------------------------------------------------------

target_include_directories(ippl
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Install Interaction-related headers
install(FILES
        ParticleInteractionBase.h
        TruncatedGreenParticleInteraction.h
        TruncatedGreenParticleInteraction.hpp
        DESTINATION include/Interaction
)
