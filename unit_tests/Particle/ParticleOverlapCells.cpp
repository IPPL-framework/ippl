#include "Ippl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "Interaction/TruncatedGreenParticleInteraction.h"
#include "Particle/ParticleSpatialOverlapLayout.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

// Checks the cell build and its scratch reuse in ParticleSpatialOverlapLayout, and the
// particle-particle field after full updates against a direct sum. Runs on any number of ranks
// (decomposition along z); CTest uses one, run `mpirun -n 2` or `-n 4` by hand to exercise the
// particle exchange.

template <typename T, typename Space>
struct InspectOverlap
    : ippl::ParticleSpatialOverlapLayout<T, 3, ippl::UniformCartesian<T, 3>, Space> {
    using Base = ippl::ParticleSpatialOverlapLayout<T, 3, ippl::UniformCartesian<T, 3>, Space>;
    using Base::Base;
    using Base::cellIndex_m;
    using Base::cellParticleCount_m;
    using Base::cellPermutationForward_m;
    using Base::cellStartingIdx_m;
    using Base::numLocalCells_m;
    using Base::particlePermutation_m;
    using Base::sendIndices_m;
    using Base::sendOffsets_m;
};

template <typename T, typename Space>
struct OverlapParticles : ippl::ParticleBase<InspectOverlap<T, Space>> {
    explicit OverlapParticles(InspectOverlap<T, Space>& layout)
        : ippl::ParticleBase<InspectOverlap<T, Space>>(layout) {
        this->addAttribute(tag);
        this->addAttribute(charge);
        this->addAttribute(field);
        this->setParticleBC(ippl::BC::NO);
    }
    ippl::ParticleAttrib<int, Space> tag;
    ippl::ParticleAttrib<T, Space> charge;
    ippl::ParticleAttrib<ippl::Vector<T, 3>, Space> field;
};

template <typename>
class ParticleOverlapCellsTest;

template <typename T, typename Space>
class ParticleOverlapCellsTest<Parameters<T, Space>> : public ::testing::Test {
public:
    using Vector    = ippl::Vector<T, 3>;
    using Mesh      = ippl::UniformCartesian<T, 3>;
    using Layout    = InspectOverlap<T, Space>;
    using Particles = OverlapParticles<T, Space>;

    void checkRebuilds() {
        const int nRanks = ippl::Comm->size();
        ippl::NDIndex<3> domain(ippl::Index(4), ippl::Index(4), ippl::Index(4 * nRanks));
        ippl::FieldLayout<3> fields(MPI_COMM_WORLD, domain, {false, false, true});
        Mesh mesh(domain, Vector(1), Vector(0));
        Layout layout(fields, mesh, T(0.5));
        Particles particles(layout);
        for (const auto n : {0u, 1u, 37u, 2049u, 9u, 0u}) {
            particles.setLocalNum(0);
            particles.create(n);
            auto r            = particles.R.getHostMirror();
            auto tags         = particles.tag.getHostMirror();
            auto q            = particles.charge.getHostMirror();
            const auto region = layout.getRegionLayout().gethLocalRegions()(ippl::Comm->rank());
            std::vector<Vector> original(n);
            std::size_t owned = 0;
            for (unsigned i = 0; i < n; ++i) {
                for (unsigned d = 0; d < 3; ++d) {
                    r(i)[d] = region[d].min() + T(0.1)
                              + T((i * (2 * d + 1)) % 17) / T(20) * region[d].length();
                }
                if (i % 7 == 0)
                    r(i)[0] = region[0].min() - T(0.2);
                else
                    ++owned;
                tags(i)     = i;
                q(i)        = T(2 * i + 1);
                original[i] = r(i);
            }
            Kokkos::deep_copy(particles.R.getView(), r);
            Kokkos::deep_copy(particles.tag.getView(), tags);
            Kokkos::deep_copy(particles.charge.getView(), q);
            layout.buildCells(particles);
            EXPECT_EQ(particles.getLocalNum(), owned);
            auto starts =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), layout.cellStartingIdx_m);
            auto counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                              layout.cellParticleCount_m);
            EXPECT_EQ(starts(0), 0);
            EXPECT_EQ(starts(starts.extent(0) - 1), n);
            EXPECT_EQ(starts(layout.numLocalCells_m), owned);
            for (std::size_t c = 0; c < counts.extent(0); ++c) {
                EXPECT_EQ(starts(c + 1) - starts(c), counts(c));
            }
            particles.setLocalNum(n);
            r    = particles.R.getHostMirror();
            tags = particles.tag.getHostMirror();
            q    = particles.charge.getHostMirror();
            Kokkos::deep_copy(r, particles.R.getView());
            Kokkos::deep_copy(tags, particles.tag.getView());
            Kokkos::deep_copy(q, particles.charge.getView());
            std::vector<int> seen(n, 0);
            for (unsigned i = 0; i < n; ++i) {
                ASSERT_GE(tags(i), 0);
                ASSERT_LT(tags(i), n);
                ++seen[tags(i)];
                EXPECT_EQ(q(i), T(2 * tags(i) + 1));
                for (unsigned d = 0; d < 3; ++d)
                    EXPECT_EQ(r(i)[d], original[tags(i)][d]);
            }
            for (auto count : seen)
                EXPECT_EQ(count, 1);
            // A new region with the same cell counts (4.04 / 0.5 -> 8 per dimension) keeps the
            // cell permutation and the particle scratch. updateLayout() only marks the cells
            // outdated; the next buildCells() rebuilds them.
            auto permutation = layout.cellPermutationForward_m;
            auto scratch     = layout.particlePermutation_m;
            mesh.setOrigin(Vector(T(0.125)));
            mesh.setMeshSpacing(Vector(T(1.01)));
            layout.updateLayout(fields, mesh);
            particles.setLocalNum(0);
            layout.buildCells(particles);
            EXPECT_EQ(layout.cellPermutationForward_m.data(), permutation.data());
            EXPECT_EQ(layout.particlePermutation_m.data(), scratch.data());
        }
        // 6 / 0.5 -> 12 cells per dimension need a new permutation.
        auto oldPermutation = layout.cellPermutationForward_m;
        mesh.setMeshSpacing(Vector(T(1.5)));
        layout.updateLayout(fields, mesh);
        particles.setLocalNum(0);
        layout.buildCells(particles);
        EXPECT_NE(layout.cellPermutationForward_m.data(), oldPermutation.data());
        auto counts =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), layout.cellParticleCount_m);
        for (std::size_t c = 0; c < counts.extent(0); ++c)
            EXPECT_EQ(counts(c), 0);
    }

    void checkField(bool periodic) {
        const int nRanks = ippl::Comm->size();
        ippl::NDIndex<3> domain(ippl::Index(4), ippl::Index(4), ippl::Index(4 * nRanks));
        ippl::FieldLayout<3> fields(MPI_COMM_WORLD, domain, {false, false, true}, periodic);
        Mesh mesh(domain, Vector(1), Vector(0));
        Layout layout(fields, mesh, T(0.5));
        Particles particles(layout);
        if (periodic)
            particles.setParticleBC(ippl::BC::PERIODIC);
        const Vector lengths{T(4), T(4), T(4 * nRanks)};
        std::vector<Vector> positions{Vector(T(2))};
        for (unsigned mask = 0; mask < 8; ++mask) {
            Vector p;
            for (unsigned d = 0; d < 3; ++d)
                p[d] = (mask & (1u << d)) ? lengths[d] - T(0.08) : T(0.08);
            positions.push_back(p);
        }
        positions.push_back(Vector{T(2.5), T(2), T(2)});
        positions.push_back(Vector{T(2.499), T(2), T(2)});
        positions.push_back(Vector{T(2), T(2), T(2)});
        positions.push_back(Vector{T(2.001), T(2), T(2)});
        const auto n = ippl::Comm->rank() == 0 ? positions.size() : 0;
        particles.create(n);
        auto r    = particles.R.getHostMirror();
        auto q    = particles.charge.getHostMirror();
        auto tags = particles.tag.getHostMirror();
        for (std::size_t i = 0; i < n; ++i) {
            r(i)    = positions[i];
            q(i)    = T(i + 1) / T(4);
            tags(i) = i;
        }
        Kokkos::deep_copy(particles.R.getView(), r);
        Kokkos::deep_copy(particles.charge.getView(), q);
        Kokkos::deep_copy(particles.tag.getView(), tags);
        for (unsigned rebuild = 0; rebuild < 2; ++rebuild) {
            layout.updateLayout(fields, mesh);
            particles.update();
            std::size_t globalOwned = 0;
            const auto localOwned   = particles.getLocalNum();
            ippl::Comm->allreduce(localOwned, globalOwned, 1, std::plus<std::size_t>());
            ASSERT_EQ(globalOwned, positions.size());
            tags = particles.tag.getHostMirror();
            Kokkos::deep_copy(tags, particles.tag.getView());
            particles.field = Vector(T(0.25));
            ippl::ParameterList parameters;
            parameters.add("rcut", T(0.5));
            parameters.add("alpha", T(4));
            parameters.add("force_constant", T(1));
            parameters.add("regularization_cutoff", T(0.01));
            using Interaction =
                ippl::TruncatedGreenParticleInteraction<Particles, decltype(particles.field),
                                                        decltype(particles.charge)>;
            Interaction interaction(particles, particles.field, particles.R, particles.charge,
                                    parameters);
            interaction.solve();
            auto actual = particles.field.getHostMirror();
            Kokkos::deep_copy(actual, particles.field.getView());
            for (std::size_t i = 0; i < localOwned; ++i) {
                ASSERT_GE(tags(i), 0);
                ASSERT_LT(tags(i), positions.size());
                Vector expected(T(0.25)), magnitude(T(1));
                for (std::size_t j = 0; j < positions.size(); ++j) {
                    Vector dist = positions[tags(i)] - positions[j];
                    for (unsigned d = 0; d < 3 && periodic; ++d) {
                        dist[d] -= std::round(dist[d] / lengths[d]) * lengths[d];
                    }
                    const T r2 = dist.dot(dist);
                    if (r2 == 0 || r2 >= T(0.25))
                        continue;
                    const T radius     = std::sqrt(r2);
                    const T correction = (std::erf(T(4) * radius)
                                          - T(8) * radius * std::exp(-T(16) * r2)
                                                / std::sqrt(Kokkos::numbers::pi_v<T>))
                                         / (r2 * radius);
                    const T coefficient       = radius < T(0.01)
                                                    ? T(1) / (T(0.01) * T(0.01) * T(0.01)) - correction
                                                    : T(1) / (r2 * radius) - correction;
                    const Vector contribution = dist * (coefficient * T(j + 1) / T(4));
                    expected -= contribution;
                    for (unsigned d = 0; d < 3; ++d)
                        magnitude[d] += std::abs(contribution[d]);
                }
                for (unsigned d = 0; d < 3; ++d) {
                    EXPECT_NEAR(actual(i)[d], expected[d],
                                T(256) * std::numeric_limits<T>::epsilon() * magnitude[d])
                        << "rebuild=" << rebuild << " particle=" << tags(i) << " dimension=" << d;
                }
            }
        }
    }
};

using Tests = TestParams::tests<>;
TYPED_TEST_SUITE(ParticleOverlapCellsTest, Tests);
TYPED_TEST(ParticleOverlapCellsTest, RebuildsAndGeometry) {
    this->checkRebuilds();
}
TYPED_TEST(ParticleOverlapCellsTest, OpenFieldMatchesDirectSum) {
    this->checkField(false);
}
TYPED_TEST(ParticleOverlapCellsTest, PeriodicFieldMatchesDirectSum) {
    this->checkField(true);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ippl::Comm->setDefaultOverallocation(1.0);
    int result;
    {
        ::testing::InitGoogleTest(&argc, argv);
        result = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return result;
}
