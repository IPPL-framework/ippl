#ifndef IPPL_P3M_PARTICLE_CONTAINER
#define IPPL_P3M_PARTICLE_CONTAINER

#include <memory>
#include "Manager/BaseManager.h"
#include "datatypes.h"
#include <initializer_list>
#include <array>
#include "Particle/ParticleSpatialOverlapLayout.h"

/**
 * @class P3MParticleContainer
 * @brief Particle Container Class for running P3M Simulations (in 3D).
 * 
 * @tparam T    The data type for simulation variables
 * @tparam Dim  The dimensionality of the simulation
*/

template<typename T, unsigned Dim = 3>
class P3MParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialOverlapLayout<T, Dim> > {
public:
    template<typename U, unsigned dim>
    using PLayout_t = ippl::ParticleSpatialOverlapLayout<U, dim, Mesh_t<dim> >;

    using Base = ippl::ParticleBase<PLayout_t<T, Dim>>;
    using Vector = ippl::Vector<T, Dim>;

    using particle_neighbor_list_type = typename PLayout_t<T, Dim>::particle_neighbor_list_type;

public:
    ippl::ParticleAttrib<T> Q; // charge
    typename Base::particle_position_type P; // particle velocity
    typename Base::particle_position_type E; // electric field at particle position
    // ippl::ParticleAttrib<double> phi;  // electric potential at particle position
    // typename Base::particle_index_type ID;      // particle global index
    // typename Base::particle_position_type F_sr; // short-range interaction force

private:
    PLayout_t<T, Dim> pl_m; // Particle layout

public:
    P3MParticleContainer(Mesh_t<Dim> &mesh, FieldLayout_t<Dim> &FL, const T &rcutoff) : pl_m(FL, mesh, rcutoff) {
        this->initialize(pl_m);

        registerAttributes();
        setBCAllPeriodic();
    }

    ~P3MParticleContainer() = default;

private:
    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }

    void registerAttributes() {
        // register the particle attributes
        this->addAttribute(Q);
        this->addAttribute(P);
        this->addAttribute(E);
        // this->addAttribute(phi);
        // this->addAttribute(ID);
        // this->addAttribute(F_sr);
    }
};

#endif
