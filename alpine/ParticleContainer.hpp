#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

    // Define the ParticlesContainer class
    template <class PLayout, typename T, unsigned Dim = 3>
    class ParticleContainer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

    public:
        ParticleAttrib<double> q;                 // charge
        typename Base::particle_position_type P;  // particle velocity
        typename Base::particle_position_type E;  // electric field at particle position
        ParticleContainer(PLayout& pl)
        : Base(pl) {
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
