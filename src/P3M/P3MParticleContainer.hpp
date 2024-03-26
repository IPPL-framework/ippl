#ifndef IPPL_P3M_PARTICLE_CONTAINER
#define IPPL_P3M_PARTICLE_CONTAINER

#include <memory>
#include "Manager/BaseManager.h"

/**
 * @class P3MParticleContainer
 * @brief Particle Container Class for running P3M Simulations (in 3D).
 * 
 * @tparam T    The data type for simulation variables
 * @tparam Dim  The dimensionality of the simulation
*/

template<typename T, unsigned Dim=3>
class P3MParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim> >{
    
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using Vector = ippl::Vector<T, Dim>;
    typedef ippl::Vector<double, Dim> Vector_t; 
    
    public:
        ippl::ParticleAttrib<double> Q;             // charge
        typename Base::particle_position_type P;    // particle velocity
        typename Base::particle_position_type E;    // electric field at particle position
        typename Base::particle_index_type ID;      // particle global index
        typename Base::particle_position_type F_sr; // short-range interaction force

    private:
        PLayout_t<T, Dim> pl_m;               

    public:
        P3MParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL) : pl_m(FL, mesh) {
            this->initialize(pl_m);
            registerAttributes();
            setBCAllPeriodic();
        }

        ~P3MParticleContainer() {}

    private:
        void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }

        void registerAttributes() {
            // register the particle attributes
            this->addAttribute(Q);
            this->addAttribute(P);
            this->addAttribute(E);
            this->addAttribute(ID);
            this->addAttribute(F_sr);
        }
             
};

#endif
