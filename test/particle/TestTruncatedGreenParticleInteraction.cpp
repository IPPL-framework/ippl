//
// TestTruncatedGreenParticleInteraction
// This program tests the short-range Ewald particle interaction using two unit-charge particles in
// a periodic ParticleSpatialOverlapLayout. It verifies that zero-distance pairs produce a finite
// zero field with the default and an updated positive regularization cutoff, and that a
// non-positive regularization cutoff is rejected. Pair scenarios check open, periodic, and
// rank-internal pairs, a cutoff changed between updates, and periodic ghosts of particles that
// are not the first ones in the container. An interaction cutoff above the layout cutoff and a
// pair loop after a rejected updateLayout() must both throw. In the moving-source mode, two
// particles with different charges and momenta must receive the Lorentz-transformed rest-frame
// field of their partner, also when the partner is a ghost on the other rank.
//
//   Usage:
//     srun -n 2 ./TestTruncatedGreenParticleInteraction --info 5
//
//

#include "Ippl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

#include "Utility/IpplException.h"

#include "Interaction/TruncatedGreenParticleInteraction.h"
#include "Particle/ParticleSpatialOverlapLayout.h"

namespace test_truncated_green_particle_interaction {
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

        TestParticles(ParticleLayout_t& layout, ippl::BC bc, bool allAxesPeriodic = false)
            : Base(layout) {
            this->addAttribute(Q);
            this->addAttribute(E);
            this->addAttribute(P);
            this->addAttribute(B);
            if (bc == ippl::BC::PERIODIC && !allAxesPeriodic) {
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
        typename Base::particle_position_type P;
        typename Base::particle_position_type B;
    };

    // A positive constructionRcut builds the layout with that cutoff and switches to rcut after a
    // first update, with setCutoff() or the three-argument updateLayout(); the pair loop must
    // refuse to run until the next update. Returns NaN if it ran with outdated cells.
    // leadingInteriorParticle puts a partnerless particle at the center before the pair, so the
    // pair particles are not the first ones copied as periodic ghosts.
    double runPairScenario(ippl::BC particleBC, bool fieldPeriodic, Scalar_t x0, Scalar_t x1,
                           Scalar_t rcut, Scalar_t alpha, Scalar_t constructionRcut = 0,
                           bool viaUpdateLayout = false, bool leadingInteriorParticle = false,
                           bool allAxesPeriodic = false) {
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
        const bool resized = constructionRcut > 0;
        ParticleLayout_t particleLayout(fieldLayout, mesh, resized ? constructionRcut : rcut);
        TestParticles particles(particleLayout, particleBC, allAxesPeriodic);

        const std::size_t first      = leadingInteriorParticle ? 1 : 0;
        const std::size_t localCount = ippl::Comm->rank() == 0 ? first + 2 : 0;
        particles.create(localCount);
        if (localCount != 0) {
            auto positions = particles.R.getHostMirror();
            auto charges   = particles.Q.getHostMirror();
            for (std::size_t i = 0; i < localCount; ++i) {
                charges(i) = 1.0;
            }
            if (leadingInteriorParticle) {
                positions(0) = Vector_t(0.5);
            }
            positions(first)     = Vector_t{x0, 0.5, 0.5};
            positions(first + 1) = Vector_t{x1, 0.5, 0.5};
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
        if (resized) {
            if (viaUpdateLayout) {
                particleLayout.updateLayout(fieldLayout, mesh, rcut);
            } else {
                particleLayout.setCutoff(rcut);
            }
            bool outdatedRejected = false;
            try {
                interaction.solve();
            } catch (const IpplException&) {
                outdatedRejected = true;
            }
            if (!outdatedRejected || particleLayout.getCutoff() != rcut) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            particles.update();
            particles.E = Vector_t(0.0);
        }
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

    // Lab-frame E and B at the lab separation r from a source with charge q and normalized
    // momentum u: the static short-range field (force_constant 1) at the rest-frame separation,
    // Lorentz-transformed to the lab. Zero once the rest-frame distance reaches rcut.
    void movingSourceReference(const Vector_t& r, const Vector_t& u, Scalar_t q, Scalar_t rcut,
                               Scalar_t alpha, Scalar_t c, Vector_t& E, Vector_t& B) {
        const Scalar_t gamma = std::sqrt(1.0 + u.dot(u));
        const Vector_t rest  = r + (u.dot(r) / (gamma + 1.0)) * u;
        const Scalar_t s     = std::sqrt(rest.dot(rest));
        E                    = Vector_t(0.0);
        B                    = Vector_t(0.0);
        if (s >= rcut) {
            return;
        }
        const Scalar_t radial = (std::erfc(alpha * s)
                                 + 2.0 * alpha * s * std::exp(-alpha * alpha * s * s)
                                       / std::sqrt(Kokkos::numbers::pi_v<Scalar_t>))
                                / (s * s * s);
        const Vector_t restField = -q * radial * rest;
        E                        = gamma * restField - (u.dot(restField) / (gamma + 1.0)) * u;
        B                        = (1.0 / (gamma * c)) * ippl::cross(u, E);
    }

    // A zero reference requires an exactly zero value.
    double relativeError(const Vector_t& value, const Vector_t& expected) {
        const Vector_t difference = value - expected;
        const double error        = std::sqrt(difference.dot(difference));
        const double scale        = std::sqrt(expected.dot(expected));
        return scale > 0.0 ? error / scale : error;
    }

    // Two particles with charges 1 and 3 and momenta u0, u1 at p0 (x < 0.5) and p1 (x > 0.5). On
    // two ranks the field layout splits x at 0.5, so each rank owns one particle and sees the
    // other only as a ghost, whose charge and momentum must be used. Returns the largest relative
    // error of E and B against movingSourceReference() over both particles.
    double runMovingSourceScenario(const Vector_t& p0, const Vector_t& p1, const Vector_t& u0,
                                   const Vector_t& u1, Scalar_t rcut, Scalar_t alpha, Scalar_t c) {
        const ippl::Vector<int, Dim> nr = {8, 8, 8};
        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(nr[d]);
        }
        const std::array<bool, Dim> isParallel = {true, false, false};
        TestFieldLayout_t fieldLayout(MPI_COMM_WORLD, owned, isParallel, false);
        const Vector_t hr(0.125);
        const Vector_t origin(0.0);
        TestMesh_t mesh(owned, hr, origin);
        ParticleLayout_t particleLayout(fieldLayout, mesh, rcut);
        TestParticles particles(particleLayout, ippl::BC::NO);

        const std::array<Vector_t, 2> positions = {p0, p1};
        const std::array<Vector_t, 2> momenta   = {u0, u1};
        const std::array<Scalar_t, 2> charges   = {1.0, 3.0};
        const std::size_t localCount            = ippl::Comm->rank() == 0 ? 2 : 0;
        particles.create(localCount);
        if (localCount != 0) {
            auto R = particles.R.getHostMirror();
            auto P = particles.P.getHostMirror();
            auto Q = particles.Q.getHostMirror();
            for (std::size_t i = 0; i < localCount; ++i) {
                R(i) = positions[i];
                P(i) = momenta[i];
                Q(i) = charges[i];
            }
            Kokkos::deep_copy(particles.R.getView(), R);
            Kokkos::deep_copy(particles.P.getView(), P);
            Kokkos::deep_copy(particles.Q.getView(), Q);
        }
        particles.update();
        particles.E = Vector_t(0.0);
        particles.B = Vector_t(0.0);

        ippl::ParameterList params;
        params.add("rcut", rcut);
        params.add("alpha", alpha);
        params.add("force_constant", 1.0);
        params.add("speed_of_light", c);

        using Interaction_t = ippl::TruncatedGreenParticleInteraction<
            TestParticles, TestParticles::particle_position_type, ippl::ParticleAttrib<Scalar_t>>;
        Interaction_t interaction(particles, particles.E, particles.B, particles.R, particles.P,
                                  particles.Q, params);
        interaction.solve();

        auto R = particles.R.getHostMirror();
        auto E = particles.E.getHostMirror();
        auto B = particles.B.getHostMirror();
        Kokkos::deep_copy(R, particles.R.getView());
        Kokkos::deep_copy(E, particles.E.getView());
        Kokkos::deep_copy(B, particles.B.getView());
        double worstLocal = 0.0;
        for (std::size_t i = 0; i < particles.getLocalNum(); ++i) {
            const std::size_t self    = R(i)[0] < 0.5 ? 0 : 1;
            const std::size_t partner = 1 - self;
            Vector_t expectedE, expectedB;
            movingSourceReference(positions[self] - positions[partner], momenta[partner],
                                  charges[partner], rcut, alpha, c, expectedE, expectedB);
            worstLocal = std::max(
                {worstLocal, relativeError(E(i), expectedE), relativeError(B(i), expectedB)});
        }
        double worst = 0.0;
        ippl::Comm->allreduce(worstLocal, worst, 1, std::greater<double>());
        return worst;
    }
}  // namespace test_truncated_green_particle_interaction

using namespace test_truncated_green_particle_interaction;

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
        // Built with a cutoff below the pair distance, then widened: the periodic halo and the
        // cells must follow the new cutoff.
        const double resizedPeriodicNorm =
            runPairScenario(ippl::BC::PERIODIC, true, 0.05, 0.95, pairRcut, pairAlpha, 0.05);
        const double relayoutPeriodicNorm =
            runPairScenario(ippl::BC::PERIODIC, true, 0.05, 0.95, pairRcut, pairAlpha, 0.05, true);
        // The periodic ghosts must be images of the boundary particles, not of the first ones.
        const double leadingInteriorPeriodicNorm = runPairScenario(
            ippl::BC::PERIODIC, true, 0.05, 0.95, pairRcut, pairAlpha, 0, false, true);
        // With every axis periodic, a pair across one face must see the image shifted only along
        // that axis.
        const double allAxesPeriodicNorm = runPairScenario(
            ippl::BC::PERIODIC, true, 0.05, 0.95, pairRcut, pairAlpha, 0, false, false, true);

        if (ippl::Comm->rank() == 0) {
            std::cout << "Pair norms: open-global=" << openBoundaryNorm
                      << ", periodic-global=" << periodicBoundaryNorm
                      << ", open-internal=" << openInternalBoundaryNorm
                      << ", resized-periodic=" << resizedPeriodicNorm
                      << ", relayout-periodic=" << relayoutPeriodicNorm
                      << ", leading-interior-periodic=" << leadingInteriorPeriodicNorm
                      << ", all-axes-periodic=" << allAxesPeriodicNorm
                      << ", expected=" << expectedNorm << std::endl;
        }

        if (openBoundaryNorm != 0.0
            || std::abs(periodicBoundaryNorm - expectedNorm) / expectedNorm > 1.0e-12
            || std::abs(openInternalBoundaryNorm - expectedNorm) / expectedNorm > 1.0e-12
            || !(std::abs(resizedPeriodicNorm - expectedNorm) / expectedNorm <= 1.0e-12)
            || !(std::abs(relayoutPeriodicNorm - expectedNorm) / expectedNorm <= 1.0e-12)
            || !(std::abs(leadingInteriorPeriodicNorm - expectedNorm) / expectedNorm <= 1.0e-12)
            || !(std::abs(allAxesPeriodicNorm - expectedNorm) / expectedNorm <= 1.0e-12)) {
            status = 1;
        }

        // Moving sources. First both partners lie inside each other's rest-frame cutoff. Then the
        // separation is 0.15 along x: the source moving along x has a rest-frame distance of 0.212
        // beyond rcut, so only the source moving along y contributes.
        const double movingBothError = runMovingSourceScenario(
            Vector_t{0.45, 0.50, 0.50}, Vector_t{0.55, 0.53, 0.48}, Vector_t{1.2, 0.3, -0.1},
            Vector_t{0.4, 0.0, 0.2}, pairRcut, pairAlpha, 2.0);
        const double movingOneError = runMovingSourceScenario(
            Vector_t{0.43, 0.50, 0.50}, Vector_t{0.58, 0.50, 0.50}, Vector_t{1.0, 0.0, 0.0},
            Vector_t{0.0, 0.5, 0.0}, pairRcut, pairAlpha, 2.0);
        if (ippl::Comm->rank() == 0) {
            std::cout << "Moving-source errors: both partners=" << movingBothError
                      << ", one partner=" << movingOneError << std::endl;
        }
        if (!(movingBothError <= 1.0e-12) || !(movingOneError <= 1.0e-12)) {
            status = 1;
        }

        // Invalid cutoffs are rejected: non-positive, or wider than half a local region.
        for (const Scalar_t invalidRcut : {Scalar_t(0), Scalar_t(0.6)}) {
            bool rejected = false;
            try {
                particleLayout.setCutoff(invalidRcut);
            } catch (const IpplException&) {
                rejected = true;
            }
            if (!rejected) {
                status = 1;
            }
        }

        ippl::ParameterList validParams;
        validParams.add("rcut", rcut);
        validParams.add("alpha", 8.0);
        validParams.add("force_constant", 1.0);

        // The cells only find partners within the layout cutoff, so a larger interaction cutoff
        // is rejected.
        ippl::ParameterList wideParams = validParams;
        wideParams.update("rcut", rcut + Scalar_t(0.05));
        Interaction_t wideInteraction(particles, particles.E, particles.R, particles.Q, wideParams);
        bool wideRejected = false;
        try {
            wideInteraction.solve();
        } catch (const IpplException&) {
            wideRejected = true;
        }
        if (!wideRejected) {
            status = 1;
        }

        // updateLayout() moves the regions before it rejects the cutoff, so the cells are outdated
        // and the pair loop must not run until the next update().
        bool relayoutRejected = false;
        try {
            particleLayout.updateLayout(fieldLayout, mesh, Scalar_t(0.6));
        } catch (const IpplException&) {
            relayoutRejected = true;
        }
        Interaction_t staleInteraction(particles, particles.E, particles.R, particles.Q,
                                       validParams);
        bool staleRejected = false;
        try {
            staleInteraction.solve();
        } catch (const IpplException&) {
            staleRejected = true;
        }
        if (!relayoutRejected || !staleRejected) {
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
