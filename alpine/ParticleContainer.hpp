#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>

#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim = 3>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>> {
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

public:
    ippl::ParticleAttrib<double> q;           // charge
    typename Base::particle_position_type P;  // particle velocity
    typename Base::particle_position_type E;  // electric field at particle position
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
        //only needed for vis
        P.set_name("velocity");
        q.set_name("charge");
        E.set_name("electric_field");
        // register the particle attributes
        this->addAttribute(q);
        this->addAttribute(P);
        this->addAttribute(E);
    }
    void setupBCs() { setBCAllPeriodic(); }

private:
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif
