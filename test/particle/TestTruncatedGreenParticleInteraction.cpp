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

#include "Particle/ParticleSpatialOverlapLayout.h"

namespace {
    constexpr unsigned Dim  = 3;
    using Scalar_t          = double;
    using Mesh_t            = ippl::UniformCartesian<Scalar_t, Dim>;
    using FieldLayout_t     = ippl::FieldLayout<Dim>;
    using TestMesh_t        = Mesh_t;
    using TestFieldLayout_t = FieldLayout_t;
    using ParticleLayout_t  = ippl::ParticleSpatialOverlapLayout<Scalar_t, Dim, TestMesh_t>;
    using Vector_t          = ippl::Vector<Scalar_t, Dim>;

    class TestParticles : public ippl::ParticleBase<ParticleLayout_t> {
    public:
        using Base = ippl::ParticleBase<ParticleLayout_t>;

        TestParticles(ParticleLayout_t& layout, ippl::BC bc)
            : Base(layout) {
            this->addAttribute(Q);
            this->addAttribute(E);
            if (bc == ippl::BC::PERIODIC) {
                typename ParticleLayout_t::bc_container_type boundaryConditions;
                boundaryConditions.fill(ippl::BC::NO);
                boundaryConditions[0] = ippl::BC::PERIODIC;
                boundaryConditions[1] = ippl::BC::PERIODIC;
                this->setParticleBC(boundaryConditions);
            } else {
                this->setParticleBC(bc);
            }
        }

        ippl::ParticleAttrib<Scalar_t> Q;
        typename Base::particle_position_type E;
    };

    double runPairScenario(ippl::BC particleBC, bool fieldPeriodic, Scalar_t x0, Scalar_t x1,
                           Scalar_t rcut, Scalar_t alpha) {
        const ippl::Vector<int, Dim> nr = {8, 8, 8};
        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(nr[d]);
        }

        const std::array<bool, Dim> isParallel = {true, false, false};
        TestFieldLayout_t fieldLayout(MPI_COMM_WORLD, owned, isParallel, fieldPeriodic);
        const Vector_t hr(0.125);
        const Vector_t origin(0.0);
        TestMesh_t mesh(owned, hr, origin);
        ParticleLayout_t particleLayout(fieldLayout, mesh, rcut);
        TestParticles particles(particleLayout, particleBC);

        const std::size_t localCount = ippl::Comm->rank() == 0 ? 2 : 0;
        particles.create(localCount);
        if (localCount != 0) {
            auto positions = particles.R.getHostMirror();
            auto charges   = particles.Q.getHostMirror();
            positions(0)   = Vector_t{x0, 0.5, 0.5};
            positions(1)   = Vector_t{x1, 0.5, 0.5};
            charges(0)     = 1.0;
            charges(1)     = 1.0;
            Kokkos::deep_copy(particles.R.getView(), positions);
            Kokkos::deep_copy(particles.Q.getView(), charges);
        }

        particles.update();
        particles.E = Vector_t(0.0);

        ippl::ParameterList params;
        params.add("rcut", rcut);
        params.add("alpha", alpha);
        params.add("force_constant", 1.0);

        using Interaction_t = ippl::TruncatedGreenParticleInteraction<
            TestParticles, TestParticles::particle_position_type, ippl::ParticleAttrib<Scalar_t>>;
        Interaction_t interaction(particles, particles.E, particles.R, particles.Q, params);
        interaction.solve();

        const auto field    = particles.E.getView();
        const auto numLocal = particles.getLocalNum();
        double normLocal    = 0.0;
        Kokkos::parallel_reduce(
            "Pair field norm",
            Kokkos::RangePolicy<typename TestParticles::particle_position_type::execution_space>(
                0, numLocal),
            KOKKOS_LAMBDA(const size_t i, double& sum) { sum += field(i).dot(field(i)); },
            Kokkos::Sum<double>(normLocal));

        double norm = 0.0;
        ippl::Comm->allreduce(normLocal, norm, 1, std::plus<double>());
        return norm;
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ippl::Comm->setDefaultOverallocation(2.0);
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
        TestParticles particles(particleLayout, ippl::BC::PERIODIC);

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

        constexpr Scalar_t pairRcut     = 0.2;
        constexpr Scalar_t pairAlpha    = 8.0;
        constexpr Scalar_t pairDistance = 0.1;
        const Scalar_t pairMagnitude =
            2.0 * pairAlpha * std::exp(-pairAlpha * pairAlpha * pairDistance * pairDistance)
                / (std::sqrt(Kokkos::numbers::pi_v<Scalar_t>) * pairDistance)
            + std::erfc(pairAlpha * pairDistance) / (pairDistance * pairDistance);
        const Scalar_t expectedNorm = 2.0 * pairMagnitude * pairMagnitude;

        const double openBoundaryNorm =
            runPairScenario(ippl::BC::NO, false, 0.05, 0.95, pairRcut, pairAlpha);
        const double periodicBoundaryNorm =
            runPairScenario(ippl::BC::PERIODIC, true, 0.05, 0.95, pairRcut, pairAlpha);
        const double openInternalBoundaryNorm =
            runPairScenario(ippl::BC::NO, false, 0.45, 0.55, pairRcut, pairAlpha);

        if (ippl::Comm->rank() == 0) {
            std::cout << "Pair norms: open-global=" << openBoundaryNorm
                      << ", periodic-global=" << periodicBoundaryNorm
                      << ", open-internal=" << openInternalBoundaryNorm
                      << ", expected=" << expectedNorm << std::endl;
        }

        if (openBoundaryNorm != 0.0
            || std::abs(periodicBoundaryNorm - expectedNorm) / expectedNorm > 1.0e-12
            || std::abs(openInternalBoundaryNorm - expectedNorm) / expectedNorm > 1.0e-12) {
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
