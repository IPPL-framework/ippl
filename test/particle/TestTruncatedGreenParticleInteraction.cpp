//
// TestTruncatedGreenParticleInteraction
// This program tests the short-range Ewald particle interaction using two unit-charge particles in
// a periodic ParticleSpatialOverlapLayout. It verifies that zero-distance pairs produce a finite
// zero field with the default and an updated positive regularization cutoff, and that a
// non-positive regularization cutoff is rejected.
//
//   Usage:
//     srun -n 2 ./TestTruncatedGreenParticleInteraction --info 5
//
//

#include "Ippl.h"

#include <cmath>
#include <iostream>

#include "Utility/IpplException.h"

#include "Interaction/TruncatedGreenParticleInteraction.h"

// ParticleSpatialOverlapLayout currently includes alpine/ParticleContainer.hpp, which expects
// these application aliases to be visible first.
template <unsigned D>
using Mesh_t = ippl::UniformCartesian<double, D>;
template <typename T, unsigned D>
using PLayout_t = ippl::ParticleSpatialLayout<T, D, Mesh_t<D>>;
template <unsigned D>
using FieldLayout_t = ippl::FieldLayout<D>;

#include "Particle/ParticleSpatialOverlapLayout.h"

namespace {
    constexpr unsigned Dim  = 3;
    using Scalar_t          = double;
    using TestMesh_t        = Mesh_t<Dim>;
    using TestFieldLayout_t = FieldLayout_t<Dim>;
    using ParticleLayout_t  = ippl::ParticleSpatialOverlapLayout<Scalar_t, Dim, TestMesh_t>;
    using Vector_t          = ippl::Vector<Scalar_t, Dim>;

    class TestParticles : public ippl::ParticleBase<ParticleLayout_t> {
    public:
        using Base = ippl::ParticleBase<ParticleLayout_t>;

        explicit TestParticles(ParticleLayout_t& layout)
            : Base(layout) {
            this->addAttribute(Q);
            this->addAttribute(E);
            this->setParticleBC(ippl::BC::PERIODIC);
        }

        ippl::ParticleAttrib<Scalar_t> Q;
        typename Base::particle_position_type E;
    };
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int status = 0;
    {
        const ippl::Vector<int, Dim> nr = {8, 8, 8};
        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(nr[d]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        TestFieldLayout_t fieldLayout(MPI_COMM_WORLD, owned, isParallel);

        const Vector_t hr(0.125);
        const Vector_t origin(0.0);
        TestMesh_t mesh(owned, hr, origin);
        constexpr Scalar_t rcut = 0.25;
        ParticleLayout_t particleLayout(fieldLayout, mesh, rcut);
        TestParticles particles(particleLayout);

        const std::size_t localCount = ippl::Comm->rank() == 0 ? 2 : 0;
        particles.create(localCount);
        if (localCount != 0) {
            auto positions = particles.R.getHostMirror();
            auto charges   = particles.Q.getHostMirror();
            for (std::size_t i = 0; i < localCount; ++i) {
                positions(i) = Vector_t(0.5);
                charges(i)   = 1.0;
            }
            Kokkos::deep_copy(particles.R.getView(), positions);
            Kokkos::deep_copy(particles.Q.getView(), charges);
        }

        particles.update();
        particles.E = Vector_t(0.0);

        ippl::ParameterList params;
        params.add("rcut", rcut);
        params.add("alpha", 8.0);
        params.add("force_constant", 1.0);

        using Interaction_t = ippl::TruncatedGreenParticleInteraction<
            TestParticles, TestParticles::particle_position_type, ippl::ParticleAttrib<Scalar_t>>;
        Interaction_t interaction(particles, particles.E, particles.R, particles.Q, params);
        interaction.solve();

        interaction.updateParameter("regularization_cutoff", 1.0e-4);
        interaction.solve();

        auto field = particles.E.getHostMirror();
        Kokkos::deep_copy(field, particles.E.getView());
        for (std::size_t i = 0; i < particles.getLocalNum(); ++i) {
            for (unsigned d = 0; d < Dim; ++d) {
                if (!std::isfinite(field(i)[d]) || field(i)[d] != 0.0) {
                    status = 1;
                }
            }
        }

        interaction.updateParameter("regularization_cutoff", -1.0);
        bool invalidCutoffRejected = false;
        try {
            interaction.solve();
        } catch (const IpplException&) {
            invalidCutoffRejected = true;
        }
        if (!invalidCutoffRejected) {
            status = 1;
        }

        int globalStatus = 0;
        ippl::Comm->allreduce(status, globalStatus, 1, std::plus<int>());
        status = globalStatus == 0 ? 0 : 1;
        if (ippl::Comm->rank() == 0 && status != 0) {
            std::cerr << "Truncated Green interaction regression failed" << std::endl;
        }
    }
    ippl::finalize();
    return status;
}
