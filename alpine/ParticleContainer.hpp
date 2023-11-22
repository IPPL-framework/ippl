#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim = 3>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>{
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

    private:
        ippl::ParticleAttrib<double> q_m;                 // charge
        typename Base::particle_position_type P_m;  // particle velocity
        typename Base::particle_position_type E_m;  // electric field at particle position
        std::shared_ptr<PLayout_t<T, Dim>> pl_m;
    public:
        ParticleContainer(std::shared_ptr<PLayout_t<T, Dim>> pl)
        : Base(*pl.get()) {
        this->initialize(*pl.get());
        registerAttributes();
        setupBCs();
        pl_m = pl;
        }

        ~ParticleContainer(){}

        ippl::ParticleAttrib<double>& getQ() { return q_m; }
        void setQ(ippl::ParticleAttrib<double>& q) { q_m = q; }

        typename Base::particle_position_type& getP() { return P_m; }
        void setP(typename Base::particle_position_type& P) { P_m = P; }

        typename Base::particle_position_type& getE() { return E_m; }
        void setE(typename Base::particle_position_type& E) { E_m = E; }

        std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; }
        void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

	void registerAttributes() {
		// register the particle attributes
		this->addAttribute(q_m);
		this->addAttribute(P_m);
		this->addAttribute(E_m);
	}
	void setupBCs() { setBCAllPeriodic(); }

    private:
       void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif
