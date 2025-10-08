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

    using particle_vector_type = typename Base::particle_position_type;
    using particle_scalar_type = ippl::ParticleAttrib<T>;
public:
    particle_scalar_type Q; // charge
    particle_vector_type P; // particle velocity
    particle_vector_type E; // electric field at particle position

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
    }
};

#endif
