#ifndef IPPL_BENCH_PARTICLES_H
#define IPPL_BENCH_PARTICLES_H

#include "Ippl.h"

#include <stdexcept>
#include <string>

namespace bench {

    constexpr unsigned Dim = 3;

    using PLayout_t     = ippl::ParticleSpatialLayout<double, Dim>;
    using Mesh_t        = ippl::UniformCartesian<double, Dim>;
    using FieldLayout_t = ippl::FieldLayout<Dim>;
    using Vector_t      = ippl::Vector<double, Dim>;

    template <typename T>
    using ParticleAttrib = ippl::ParticleAttrib<T>;

    template <class PLayout>
    class BenchParticles : public ippl::ParticleBase<PLayout> {
    public:
        std::array<bool, Dim> isParallel_m;

        ParticleAttrib<double> qm;
        typename ippl::ParticleBase<PLayout>::particle_position_type P;
        typename ippl::ParticleBase<PLayout>::particle_position_type E;

        BenchParticles(PLayout& pl, std::array<bool, Dim> isParallel)
            : ippl::ParticleBase<PLayout>(pl)
            , isParallel_m(isParallel) {
            this->addAttribute(qm);
            this->addAttribute(P);
            this->addAttribute(E);
            this->setParticleBC(ippl::BC::PERIODIC);
        }
    };

    inline ippl::CountExchange parseMode(const std::string& s) {
        if (s == "rma")
            return ippl::CountExchange::RMA;
        if (s == "p2p")
            return ippl::CountExchange::P2P_GPU;
        if (s == "alltoall")
            return ippl::CountExchange::Alltoall_GPU;
        throw std::invalid_argument("Unknown exchange mode: " + s);
    }

}  // namespace bench

#endif
