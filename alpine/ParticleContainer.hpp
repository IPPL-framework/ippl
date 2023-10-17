#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

    // Define the ParticlesContainer class
    template <typename T, unsigned Dim = 3>
    class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>{
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

    public:
        ippl::ParticleAttrib<double> q;                 // charge
        typename Base::particle_position_type P;  // particle velocity
        typename Base::particle_position_type E;  // electric field at particle position
        ParticleContainer(ippl::ParticleSpatialLayout<T, Dim>& pl)
        : Base(pl) {
        this->initialize(pl);
        registerAttributes();
        setupBCs();
        }
	void registerAttributes() {
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
