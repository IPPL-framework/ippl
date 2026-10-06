#include "Ippl.h"

#include <algorithm>
#include <array>
#include <vector>

#include "Particle/ParticleSpatialOverlapLayout.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class ParticleOverlapLocateTest;

template <typename T, typename ExecSpace>
class ParticleOverlapLocateTest<Parameters<T, ExecSpace>> : public ::testing::Test {
public:
    using mesh_type = ippl::UniformCartesian<T, 3>;
    using layout_type = ippl::ParticleSpatialOverlapLayout<T, 3, mesh_type, ExecSpace>;
    using bunch_type = ippl::ParticleBase<layout_type>;
    using vector_type = ippl::Vector<T, 3>;

    void checkExchange(bool initiallyEmptyRanks) {
        const int rank = ippl::Comm->rank();
        const int nRanks = ippl::Comm->size();
        if (nRanks < 8) {
            GTEST_SKIP() << "Requires at least eight ranks for remote migration";
        }

        ippl::NDIndex<3> domain(ippl::Index(4), ippl::Index(4), ippl::Index(4 * nRanks));
        std::array<bool, 3> parallel{false, false, true};
        ippl::FieldLayout<3> fieldLayout(MPI_COMM_WORLD, domain, parallel);
        mesh_type mesh(domain, vector_type(T(1)), vector_type(T(0)));
        layout_type layout(fieldLayout, mesh, T(0.5));
        bunch_type bunch(layout);
        constexpr unsigned N = 8;
        bunch.create(initiallyEmptyRanks && rank % 2 != 0 ? 0 : N);

        const auto regions = layout.getRegionLayout().gethLocalRegions();
        auto center = [&](int owner) {
            vector_type position;
            for (unsigned d = 0; d < 3; ++d) {
                position[d] = (regions(owner)[d].min() + regions(owner)[d].max()) / T(2);
            }
            return position;
        };
        const int destination = (rank + 3) % nRanks;
        auto positions = bunch.R.getHostMirror();
        for (unsigned i = 0; i < bunch.getLocalNum(); ++i) {
            positions(i) = center(destination);
            positions(i)[0] += T(i) / T(32);
        }
        Kokkos::deep_copy(bunch.R.getView(), positions);

        // Exercise the production allocation of particleRanks, which direct locate calls bypass.
        layout.particleExchange(bunch);

        const int source = (rank + nRanks - 3) % nRanks;
        const unsigned expected = initiallyEmptyRanks && source % 2 != 0 ? 0 : N;
        ASSERT_EQ(bunch.getLocalNum(), expected);
        const auto received =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bunch.R.getView());
        std::vector<T> actualX;
        std::vector<T> expectedX;
        for (unsigned i = 0; i < expected; ++i) {
            actualX.push_back(received(i)[0]);
            expectedX.push_back(center(rank)[0] + T(i) / T(32));
            for (unsigned d = 1; d < 3; ++d) {
                EXPECT_EQ(received(i)[d], center(rank)[d]);
            }
        }
        std::sort(actualX.begin(), actualX.end());
        EXPECT_EQ(actualX, expectedX);
    }

    void checkDestinations(bool overlapTwoRemoteRanks) {
        const int rank = ippl::Comm->rank();
        const int nRanks = ippl::Comm->size();
        if (nRanks < 8) {
            GTEST_SKIP() << "Requires at least eight ranks for a non-neighbor overlap pair";
        }

        ippl::NDIndex<3> domain(ippl::Index(4), ippl::Index(4), ippl::Index(4 * nRanks));
        std::array<bool, 3> parallel{false, false, true};
        ippl::FieldLayout<3> fieldLayout(MPI_COMM_WORLD, domain, parallel);
        mesh_type mesh(domain, vector_type(T(1)), vector_type(T(0)));
        constexpr T cutoff = T(0.5);
        layout_type layout(fieldLayout, mesh, cutoff);
        bunch_type bunch(layout);
        constexpr unsigned N = 8;
        bunch.create(N);

        const auto regions = layout.getRegionLayout().gethLocalRegions();
        const auto neighbors = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), layout.getFlatNeighbors(fieldLayout.getNeighbors()));
        ASSERT_GT(neighbors.extent(0), 0u);
        const int neighbor = neighbors(0);

        auto isRemote = [&](int candidate) {
            if (candidate == rank) {
                return false;
            }
            for (unsigned i = 0; i < neighbors.extent(0); ++i) {
                if (candidate == neighbors(i)) {
                    return false;
                }
            }
            return true;
        };
        int remote = -1;
        int remoteNext = -1;
        for (int a = 0; a < nRanks; ++a) {
            for (int b = 0; b < nRanks; ++b) {
                if (isRemote(a) && isRemote(b)
                    && regions(a)[2].max() == regions(b)[2].min()) {
                    remote = a;
                    remoteNext = b;
                }
            }
        }
        ASSERT_GE(remote, 0);
        ASSERT_GE(remoteNext, 0);

        auto center = [&](int owner) {
            vector_type position;
            for (unsigned d = 0; d < 3; ++d) {
                position[d] = (regions(owner)[d].min() + regions(owner)[d].max()) / T(2);
            }
            return position;
        };
        auto positions = bunch.R.getHostMirror();
        std::vector<std::vector<int>> expected(N, {rank});
        for (unsigned i = 0; i < N; ++i) {
            positions(i) = center(rank);
        }
        // Only two particles leave. Their full indices (2, 7) exceed the compact list extent (2).
        positions(2) = center(neighbor);
        expected[2] = {neighbor};
        positions(7) = center(remote);
        expected[7] = {remote};
        if (overlapTwoRemoteRanks) {
            positions(7)[2] = regions(remote)[2].max();
            expected[7].push_back(remoteNext);
        }
        Kokkos::deep_copy(bunch.R.getView(), positions);

        typename layout_type::locate_type ranks("ranks", 0);
        typename layout_type::locate_type offsets("offsets", N + 1);
        typename layout_type::bool_type invalid("invalid", N);
        typename layout_type::locate_type nSends("nSends", nRanks);
        typename layout_type::locate_type destinations("destinations", nRanks);
        const auto [nInvalid, nDestinations] =
            layout.locateParticles(bunch, ranks, offsets, invalid, nSends, destinations);

        const auto hostRanks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ranks);
        const auto hostOffsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets);
        const auto hostInvalid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid);
        const auto hostSends = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nSends);
        const auto hostDestinations =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), destinations);
        EXPECT_EQ(nInvalid, 2u);
        EXPECT_EQ(nDestinations, overlapTwoRemoteRanks ? 4u : 3u);
        EXPECT_EQ(hostOffsets(0), 0);
        EXPECT_EQ(hostOffsets(N), overlapTwoRemoteRanks ? N + 1 : N);

        std::vector<unsigned> expectedSends(nRanks, 0);
        for (unsigned i = 0; i < N; ++i) {
            ASSERT_EQ(hostOffsets(i + 1) - hostOffsets(i), expected[i].size());
            ASSERT_LE(hostOffsets(i + 1), hostRanks.extent(0));
            std::vector<int> actual;
            for (auto j = hostOffsets(i); j < hostOffsets(i + 1); ++j) {
                actual.push_back(hostRanks(j));
            }
            std::sort(actual.begin(), actual.end());
            std::sort(expected[i].begin(), expected[i].end());
            EXPECT_EQ(actual, expected[i]) << "Particle " << i << " on source rank " << rank;
            EXPECT_EQ(hostInvalid(i), i == 2 || i == 7);
            for (int owner : expected[i]) {
                ++expectedSends[owner];
            }
        }
        std::vector<int> actualDestinations;
        std::vector<int> expectedDestinations;
        for (int owner = 0; owner < nRanks; ++owner) {
            EXPECT_EQ(hostSends(owner), expectedSends[owner]);
            if (expectedSends[owner] > 0) {
                expectedDestinations.push_back(owner);
            }
        }
        ASSERT_LE(nDestinations, hostDestinations.extent(0));
        for (unsigned i = 0; i < nDestinations; ++i) {
            actualDestinations.push_back(hostDestinations(i));
        }
        std::sort(actualDestinations.begin(), actualDestinations.end());
        EXPECT_EQ(actualDestinations, expectedDestinations);
    }
};

using Tests = TestParams::tests<>;
TYPED_TEST_SUITE(ParticleOverlapLocateTest, Tests);

TYPED_TEST(ParticleOverlapLocateTest, SparseOutsideIds) {
    this->checkDestinations(false);
}

TYPED_TEST(ParticleOverlapLocateTest, SparseOutsideIdsWithRemoteGhost) {
    this->checkDestinations(true);
}

TYPED_TEST(ParticleOverlapLocateTest, ExchangeRemoteParticles) {
    this->checkExchange(false);
}

TYPED_TEST(ParticleOverlapLocateTest, ExchangeWithInitiallyEmptyRanks) {
    this->checkExchange(true);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int result;
    {
        ::testing::InitGoogleTest(&argc, argv);
        result = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return result;
}
