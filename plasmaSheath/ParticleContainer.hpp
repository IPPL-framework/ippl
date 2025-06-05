#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim = 1>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>> {
    using Base                   = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using particle_velocity_type = ParticleAttrib<Vector_t<T, 3>>;

public:
    ippl::ParticleAttrib<double> q;  // charge
    ippl::ParticleAttrib<T> m;       // mass
    particle_velocity_type P;        // particle velocity
    particle_velocity_type E;        // electric field at particle position
private:
    PLayout_t<T, Dim> pl_m;

public:
    ParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
    }

    ~ParticleContainer() {}

    std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; }
    void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

    void registerAttributes() {
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(m);
        this->addAttribute(P);
        this->addAttribute(E);
    }
    void setupBCs() { setSinkBCs(); }

private:
    void setSinkBCs() { this->setParticleBC(ippl::BC::SINK); }
};

#endif
