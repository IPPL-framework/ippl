#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

/**
 * @brief Class representing a container for particles.
 * 
 * @tparam T Type of the particle attribute.
 * @tparam Dim Dimensionality of the particle space (default is 3).
 */
template <typename T, unsigned Dim = 3>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>> {
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

public:
    /**
     * @brief Mass of the particle.
     */
    ippl::ParticleAttrib<double> m;

    /**
     * @brief Velocity of the particle.
     */
    typename Base::particle_position_type V;

    /**
     * @brief Gravitational field at the particle position.
     */
    typename Base::particle_position_type F;

    /**
     * @brief Constructor for ParticleContainer.
     * 
     * @param mesh Reference to the mesh object.
     * @param FL Reference to the field layout object.
     */
    ParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
    }

    /**
     * @brief Destructor for ParticleContainer.
     */
    ~ParticleContainer() {}

    /**
     * @brief Get the particle layout.
     * 
     * @return Shared pointer to the particle layout.
     */
    std::shared_ptr<PLayout_t<T, Dim>> getPL() const { return pl_m; }

    /**
     * @brief Set the particle layout.
     * 
     * @param pl Shared pointer to the particle layout.
     */
    void setPL(const std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

    /**
     * @brief Register the particle attributes.
     */
    void registerAttributes() {
        this->addAttribute(m);
        this->addAttribute(V);
        this->addAttribute(F);
    }

    /**
     * @brief Setup boundary conditions.
     */
    void setupBCs() { setBCAllPeriodic(); }

private:
    /**
     * @brief Particle layout.
     */
    PLayout_t<T, Dim> pl_m;

    /**
     * @brief Set all boundary conditions to periodic.
     */
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif // IPPL_PARTICLE_CONTAINER_H