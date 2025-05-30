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

template<typename T, unsigned Dim=3>
class P3MParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialOverlapLayout<T, Dim> >{
    using Device = Kokkos::DefaultExecutionSpace;
    using Host = Kokkos::DefaultHostExecutionSpace;
    using NList_t = Kokkos::View<unsigned int *, Device>;
    using Offset_t = const std::array<std::array<int, Dim>, ippl::detail::countHypercubes(Dim) / 2 + 1>; // 14, 3 in 3 dimensions

    using Base = ippl::ParticleBase<ippl::ParticleSpatialOverlapLayout<T, Dim>>;
    using Vector = ippl::Vector<T, Dim>;

    template<typename U, unsigned dim>
    using PLayout_t = typename ippl::ParticleSpatialOverlapLayout<U, dim, Mesh_t<dim>>;

    public:
        ippl::ParticleAttrib<T> Q;             // charge
        typename Base::particle_position_type P;    // particle velocity
        typename Base::particle_position_type E;    // electric field at particle position
        // ippl::ParticleAttrib<double> phi;  // electric potential at particle position
        // typename Base::particle_index_type ID;      // particle global index
        // typename Base::particle_position_type F_sr; // short-range interaction force

    private:
        PLayout_t<T, Dim> pl_m;     // Particle layout
        NList_t nl_m;               // NeighborList
        bool nlValid_m;             // true if neighbor list was initialized
        bool *neighbors_m;
        // Offset_t offset_m;

    public:
        // only half of the offset because the other half is handled by due to pairwise interaction
        // always using the offsets such that |offset| >= 0 would make it possible to generalize this do more dimensions
        static constexpr Offset_t offset_m{{{1, 1, 1},
                                            {0, 1, 1},
                                            {-1, 1, 1},
                                            {1, 0, 1},
                                            {0, 0, 1},
                                            {-1, 0, 1},
                                            {1, -1, 1},
                                            {0, -1, 1},
                                            {-1, -1, 1},
                                            {1, 1, 0},
                                            {0, 1, 0},
                                            {-1, 1, 0},
                                            {1, 0, 0},
                                            {0, 0, 0}}};

    public:
        P3MParticleContainer(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL, const T& rcutoff) : pl_m(FL, mesh, rcutoff), nlValid_m(false) {
            this->initialize(pl_m);
            
            // initialize to zero
            neighbors_m = new bool[(ippl::Comm->size())];
            for (int i = 0; i < ippl::Comm->size(); ++i){
                neighbors_m[i] = false;
            }

            registerAttributes();
            setBCAllPeriodic();
        }

        ~P3MParticleContainer() {}

        void setNL(NList_t& nl) {
            this->nlValid_m = true;
            this->nl_m = nl;
        }

        NList_t& getNL() {
            if(nlValid_m){
                return nl_m;
            } else {
                throw 0;
            }
        }

        void setNeighbors(bool *neighbors_) {
            neighbors_m = neighbors_;
        }

        bool *getNeighbors() {
            return neighbors_m;
        }

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
