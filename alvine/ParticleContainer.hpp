#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim = 3>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>{
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using vorticity_type = std::conditional<Dim == 2, ippl::ParticleAttrib<T>, typename Base::particle_position_type >::type;

    public:
        typename Base::particle_position_type P;  
        vorticity_type omega;
        typename Base::particle_position_type R_old;

    private:
        PLayout_t<T, Dim> pl_m;
    public:
        ParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
        }

        ~ParticleContainer(){}

        std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; }
        void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

	void registerAttributes() {
		// register the particle attributes

		this->addAttribute(P);
        this->addAttribute(omega);
        this->addAttribute(R_old);
	}
	void setupBCs() { setBCAllPeriodic(); }

    private:
       void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif
