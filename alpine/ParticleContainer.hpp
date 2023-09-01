#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>
#include "Manager/BaseManager.h"

namespace ippl {
    // Define the ParticlesContainer class
    template <class PLayout, typename T, unsigned Dim = 3>
    class ParticleContainer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

    public:
        ParticleAttrib<double> q;                 // charge
         double Q_m;
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
}  // namespace ippl

#endif
