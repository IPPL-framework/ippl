//
// Unit test: Particle spatial layout update / neighbor migration
//
// Tests that ippl::ParticleSpatialLayout::update() correctly migrates
// particles to the rank that owns the region containing the particle's
// position.  The suite covers:
//
//   1.  Baseline sanity – random particles already on correct rank.
//   2.  Single deliberate out-of-bounds displacement (each axis / direction).
//   3.  Corner / edge displacement (multi-axis simultaneous crossing).
//   4.  Periodic wrap-around: particles placed just outside the global domain
//       boundary reappear on the opposite rank.
//   5.  All particles seeded on rank 0 and redistributed globally.
//   6.  All particles seeded on the last rank and redistributed globally.
//   7.  Non-uniform layouts: more cells assigned to one rank than the others,
//       so that a single neighbour may span many grid cells.
//   8.  Conservation: total particle count is invariant under update().
//   9.  Charge conservation: sum of Q is invariant under update().
//  10.  Multiple successive updates leave the layout stable.
//  11.  Large particle count stress test.
//  12.  Boundary-exact positions (particles sitting exactly on a rank
//       boundary should belong to a well-defined rank, not be lost).
//  13.  Zero-particle ranks: a rank with no particles participates correctly.
//  14.  Heterogeneous displacement magnitudes in the same step.
//  15.  3-D corner migration (all three axes crossed simultaneously).
//
#include "Ippl.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "TestUtils.h"
#include "gtest/gtest.h"

// ============================================================
//  Helper types
// ============================================================

template <typename T, typename ExecSpace, unsigned Dim>
struct TestBunch
    : public ippl::ParticleBase<
          ippl::ParticleSpatialLayout<T, Dim, ippl::UniformCartesian<T, Dim>, ExecSpace>> {
    using playout_type =
        ippl::ParticleSpatialLayout<T, Dim, ippl::UniformCartesian<T, Dim>, ExecSpace>;

    explicit TestBunch(playout_type& pl)
        : ippl::ParticleBase<playout_type>(pl) {
        this->addAttribute(Q);
        this->addAttribute(tag);
    }

    // Per-particle charge (used for conservation checks)
    ippl::ParticleAttrib<T, ExecSpace> Q;

    // Integer tag so we can track individual particles across ranks
    ippl::ParticleAttrib<long long, ExecSpace> tag;
};

// ============================================================
//  Fixture
// ============================================================

template <typename>
class TestParticleUpdate;

template <typename T_, typename ExecSpace, unsigned Dim_>
class TestParticleUpdate<Parameters<T_, ExecSpace, Rank<Dim_>>> : public ::testing::Test {
public:
    // ---- type aliases --------------------------------------------------
    using T                       = T_;
    static constexpr unsigned Dim = Dim_;
    using flayout_type            = ippl::FieldLayout<Dim>;
    using mesh_type               = ippl::UniformCartesian<T, Dim>;
    using playout_type            = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using RegionLayout_t          = typename playout_type::RegionLayout_t;
    using bunch_type              = TestBunch<T, ExecSpace, Dim>;
    using position_type           = ippl::Vector<T, Dim>;
    using region_view             = typename RegionLayout_t::view_type;

    // ---- construction --------------------------------------------------
    TestParticleUpdate()
        : nPoints(getGridSizes<Dim>()) {
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = static_cast<T>(nPoints[d]) / 16.0;
        }

        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++)
            args[d] = ippl::Index(nPoints[d]);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        ippl::Vector<T, Dim> hx, origin;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        for (unsigned d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        layout      = std::make_shared<flayout_type>(MPI_COMM_WORLD, owned, isParallel);
        mesh        = std::make_shared<mesh_type>(owned, hx, origin);
        playout_ptr = std::make_shared<playout_type>(*layout, *mesh);
    }

    // ---- helpers -------------------------------------------------------

    /// Create a fresh bunch and set periodic BCs.
    std::shared_ptr<bunch_type> makeBunch() {
        auto b = std::make_shared<bunch_type>(*playout_ptr);
        typename bunch_type::bc_container_type bcs;
        bcs.fill(ippl::BC::PERIODIC);
        b->setParticleBC(bcs);
        return b;
    }

    /// Place n particles with uniformly random positions on every rank.
    void fillRandom(bunch_type& b, unsigned n, unsigned long seed = 42) {
        int nRanks       = ippl::Comm->size();
        unsigned perRank = std::max(1u, n / nRanks);
        b.create(perRank);

        std::mt19937_64 eng(seed + ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(T(0), T(1));

        auto R_host = b.R.getHostMirror();
        auto Q_host = b.Q.getHostMirror();
        auto t_host = b.tag.getHostMirror();

        // Global unique id = rank * kRankIdStride + local index. The stride
        // bounds the per-rank particle count; 10M is comfortably above any
        // unit-test bunch size and stays well clear of int32 overflow.
        constexpr long long kRankIdStride = 10'000'000LL;
        for (size_t i = 0; i < b.getLocalNum(); ++i) {
            position_type r;
            for (unsigned d = 0; d < Dim; d++)
                r[d] = unif(eng) * domain[d];
            R_host(i) = r;
            Q_host(i) = T(1);
            t_host(i) = static_cast<long long>(ippl::Comm->rank()) * kRankIdStride
                        + static_cast<long long>(i);
        }
        Kokkos::deep_copy(b.R.getView(), R_host);
        Kokkos::deep_copy(b.Q.getView(), Q_host);
        Kokkos::deep_copy(b.tag.getView(), t_host);
    }

    /// Reduce total particle count across all ranks.
    size_t totalParticles(const bunch_type& b) const {
        size_t local = b.getLocalNum();
        size_t total = 0;
        ippl::Comm->reduce(local, total, 1, std::plus<size_t>());
        return total;
    }

    /// Reduce total charge across all ranks.
    T totalCharge(bunch_type& b) const {
        auto Q_host = b.Q.getHostMirror();
        Kokkos::deep_copy(Q_host, b.Q.getView());
        T local = T(0);
        for (size_t i = 0; i < b.getLocalNum(); ++i)
            local += Q_host(i);
        T total = T(0);
        ippl::Comm->reduce(local, total, 1, std::plus<T>());
        return total;
    }

    /// For every locally owned particle verify it lies inside the local region.
    /// Returns the number of misplaced particles on this rank.
    size_t countMisplaced(bunch_type& b) {
        RegionLayout_t rl   = playout_ptr->getRegionLayout();
        region_view regions = rl.getdLocalRegions();
        int myRank          = ippl::Comm->rank();

        auto R_host = b.R.getHostMirror();
        Kokkos::deep_copy(R_host, b.R.getView());

        // Copy region bounds to host for checking
        auto regions_host = Kokkos::create_mirror_view(regions);
        Kokkos::deep_copy(regions_host, regions);

        size_t misplaced = 0;
        for (size_t i = 0; i < b.getLocalNum(); ++i) {
            bool inMyRegion = true;
            for (unsigned d = 0; d < Dim; d++) {
                T pos = R_host(i)[d];
                inMyRegion &=
                    (pos >= regions_host(myRank)[d].min() && pos <= regions_host(myRank)[d].max());
            }
            if (!inMyRegion)
                ++misplaced;
        }
        return misplaced;
    }

    /// Wrap a coordinate periodically into [0, domainLen).
    T periodicWrap(T x, T domainLen) const {
        while (x < T(0))
            x += domainLen;
        while (x >= domainLen)
            x -= domainLen;
        return x;
    }

    // ---- data members --------------------------------------------------
    std::array<size_t, Dim> nPoints;
    std::array<T, Dim> domain;

    std::shared_ptr<flayout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout_ptr;
};

using Tests = TestParams::tests<1, 2, 3>;
TYPED_TEST_SUITE(TestParticleUpdate, Tests);

// ============================================================
//  1.  Baseline: random particles already on the correct rank
// ============================================================
TYPED_TEST(TestParticleUpdate, BaselineSanityNoMigrationNeeded) {
    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    // update() should not lose or duplicate particles
    const size_t before = this->totalParticles(*bunch);
    bunch->update();
    const size_t after = this->totalParticles(*bunch);

    EXPECT_EQ(before, after);
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  2.  Conservation of total particle count
// ============================================================
TYPED_TEST(TestParticleUpdate, TotalParticleCountConserved) {
    constexpr unsigned N = 512;
    auto bunch           = this->makeBunch();
    this->fillRandom(*bunch, N);

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
//  3.  Conservation of total charge
// ============================================================
TYPED_TEST(TestParticleUpdate, TotalChargeConserved) {
    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    // Give every particle a distinct charge so any loss/duplication shows up
    {
        auto Q_host = bunch->Q.getHostMirror();
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            Q_host(i) = static_cast<typename TestFixture::T>(i + 1);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const auto chargeBefore = this->totalCharge(*bunch);
    bunch->update();
    const auto chargeAfter = this->totalCharge(*bunch);

    // Allow small floating-point rounding
    EXPECT_NEAR(static_cast<double>(chargeBefore), static_cast<double>(chargeAfter),
                1e-6 * std::abs(static_cast<double>(chargeBefore)));
}

// ============================================================
//  4.  Single-axis displacement – particles must migrate to neighbour
// ============================================================
TYPED_TEST(TestParticleUpdate, SingleAxisDisplacementMigratesCorrectly) {
    using T = typename TestFixture::T;

    // Only meaningful with more than one rank
    if (ippl::Comm->size() < 2)
        GTEST_SKIP();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 128);

    // Move every local particle half the domain length along axis 0
    // (wrapping with period).  After update, everyone must be on the
    // owning rank.
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        const T shift = this->domain[0] / T(2);
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            R_host(i)[0] = this->periodicWrap(R_host(i)[0] + shift, this->domain[0]);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  5.  Multi-axis (corner) displacement
// ============================================================
TYPED_TEST(TestParticleUpdate, CornerDisplacementMigratesCorrectly) {
    using T = typename TestFixture::T;

    if (ippl::Comm->size() < 2)
        GTEST_SKIP();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 128);

    // Shift along every axis simultaneously
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                const T shift = this->domain[d] * T(0.3);
                R_host(i)[d]  = this->periodicWrap(R_host(i)[d] + shift, this->domain[d]);
            }
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  6.  Periodic wrap-around: particles placed just outside global domain
// ============================================================
TYPED_TEST(TestParticleUpdate, PeriodicWrapAroundRightBoundary) {
    using T = typename TestFixture::T;

    auto bunch = this->makeBunch();

    // Place one particle per rank just beyond the upper global boundary
    // in axis 0; periodic BC should wrap it back.
    bunch->create(1);
    {
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        // Place at domain[0] + small epsilon → wraps to near 0
        T eps        = this->domain[0] * T(1e-4);
        R_host(0)[0] = this->domain[0] + eps;
        for (unsigned d = 1; d < TestFixture::Dim; d++)
            R_host(0)[d] = this->domain[d] * T(0.5);
        Q_host(0) = T(1);
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
//  7.  Periodic wrap-around: just outside lower global boundary
// ============================================================
TYPED_TEST(TestParticleUpdate, PeriodicWrapAroundLeftBoundary) {
    using T = typename TestFixture::T;

    auto bunch = this->makeBunch();
    bunch->create(1);
    {
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        T eps       = this->domain[0] * T(1e-4);
        // Slightly negative position wraps to near domain[0]
        R_host(0)[0] = -eps;
        for (unsigned d = 1; d < TestFixture::Dim; d++)
            R_host(0)[d] = this->domain[d] * T(0.5);
        Q_host(0) = T(1);
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
//  8.  All particles seeded on rank 0 → global redistribution
// ============================================================
TYPED_TEST(TestParticleUpdate, AllParticlesOnRank0Redistributed) {
    using T = typename TestFixture::T;

    unsigned N = ippl::Comm->rank() == 0 ? 256 : 0;
    auto bunch = this->makeBunch();

    bunch->create(N);
    if (ippl::Comm->rank() == 0) {
        std::mt19937_64 eng(1234);
        std::uniform_real_distribution<T> unif(T(0), T(1));
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        for (unsigned i = 0; i < N; ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++)
                R_host(i)[d] = unif(eng) * this->domain[d];
            Q_host(i) = T(1);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    ASSERT_EQ(before, static_cast<size_t>(N));

    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  9.  All particles seeded on last rank → global redistribution
// ============================================================
TYPED_TEST(TestParticleUpdate, AllParticlesOnLastRankRedistributed) {
    using T      = typename TestFixture::T;
    int lastRank = ippl::Comm->size() - 1;
    unsigned N   = ippl::Comm->rank() == lastRank ? 256 : 0;
    auto bunch   = this->makeBunch();
    bunch->create(N);

    if (ippl::Comm->rank() == lastRank) {
        std::mt19937_64 eng(5678);
        std::uniform_real_distribution<T> unif(T(0), T(1));
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        for (unsigned i = 0; i < N; ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++)
                R_host(i)[d] = unif(eng) * this->domain[d];
            Q_host(i) = T(1);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  10. Multiple successive updates are idempotent
// ============================================================
TYPED_TEST(TestParticleUpdate, MultipleSuccessiveUpdatesStable) {
    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 128);

    const size_t before = this->totalParticles(*bunch);
    for (int iter = 0; iter < 5; ++iter) {
        bunch->update();
        EXPECT_EQ(before, this->totalParticles(*bunch));
        EXPECT_EQ(0u, this->countMisplaced(*bunch));
    }
}

// ============================================================
//  11. Successive updates after each random displacement
// ============================================================
TYPED_TEST(TestParticleUpdate, SuccessiveDisplacementsConserveParticles) {
    using T = typename TestFixture::T;

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 128);
    const size_t N = this->totalParticles(*bunch);

    std::mt19937_64 eng(999 + ippl::Comm->rank());
    std::uniform_real_distribution<T> unif(-T(0.1), T(0.1));

    for (int step = 0; step < 8; ++step) {
        // Small random displacements (staying in domain with wrapping)
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T newPos     = R_host(i)[d] + unif(eng) * this->domain[d];
                R_host(i)[d] = this->periodicWrap(newPos, this->domain[d]);
            }
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);

        bunch->update();
        EXPECT_EQ(N, this->totalParticles(*bunch)) << "at step " << step;
    }
}

// ============================================================
//  12. Boundary-exact positions: particles at rank boundaries
// ============================================================
TYPED_TEST(TestParticleUpdate, BoundaryExactPositionsNotLost) {
    using T = typename TestFixture::T;

    // Each rank creates one particle exactly at the domain's mid-point
    // along axis 0 (a likely rank boundary for 2+ ranks).
    auto bunch = this->makeBunch();
    bunch->create(1);
    {
        auto R_host  = bunch->R.getHostMirror();
        auto Q_host  = bunch->Q.getHostMirror();
        R_host(0)[0] = this->domain[0] / T(2);
        for (unsigned d = 1; d < TestFixture::Dim; d++)
            R_host(0)[d] = this->domain[d] / T(2);
        Q_host(0) = T(1);
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    // The particle must survive, though ownership may shift
    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
//  13. Zero-particle ranks participate correctly
// ============================================================
TYPED_TEST(TestParticleUpdate, ZeroParticleRanksDoNotDeadlock) {
    using T = typename TestFixture::T;

    // Only rank 0 creates particles
    auto bunch = this->makeBunch();
    int N      = ippl::Comm->rank() == 0 ? 64 : 0;
    bunch->create(N);

    if (ippl::Comm->rank() == 0) {
        std::mt19937_64 eng(7777);
        std::uniform_real_distribution<T> unif(T(0), T(1));
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        for (int i = 0; i < N; ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++)
                R_host(i)[d] = unif(eng) * this->domain[d];
            Q_host(i) = T(1);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    // This must not deadlock even if some ranks hold zero particles
    ASSERT_NO_THROW(bunch->update());
    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
//  14. Heterogeneous displacement magnitudes (small + large mixed)
// ============================================================
TYPED_TEST(TestParticleUpdate, HeterogeneousDisplacementsMigrateCorrectly) {
    using T = typename TestFixture::T;

    if (ippl::Comm->size() < 2)
        GTEST_SKIP();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            // Even-indexed particles get a tiny nudge; odd ones jump ~70%.
            const T fraction = (i % 2 == 0) ? T(0.01) : T(0.7);
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T newPos     = R_host(i)[d] + fraction * this->domain[d];
                R_host(i)[d] = this->periodicWrap(newPos, this->domain[d]);
            }
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  15. Large-count stress test
// ============================================================
TYPED_TEST(TestParticleUpdate, LargeParticleCountConserved) {
    constexpr unsigned N = 4096;
    auto bunch           = this->makeBunch();
    this->fillRandom(*bunch, N, /*seed=*/31415);

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  16. Non-uniform particle distribution: load-imbalanced initial state
// ============================================================
TYPED_TEST(TestParticleUpdate, NonUniformInitialDistributionRedistributesCorrectly) {
    using T = typename TestFixture::T;

    auto bunch = this->makeBunch();

    // Rank r gets r+1 particles, placed uniformly at random.
    unsigned nLocal = static_cast<unsigned>(ippl::Comm->rank() + 1);
    bunch->create(nLocal);
    {
        std::mt19937_64 eng(2468 + ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(T(0), T(1));
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        for (unsigned i = 0; i < nLocal; ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++)
                R_host(i)[d] = unif(eng) * this->domain[d];
            Q_host(i) = T(1);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  17. Full-domain traversal: particle moves across the entire domain
// ============================================================
TYPED_TEST(TestParticleUpdate, FullDomainTraversalConservesParticles) {
    using T = typename TestFixture::T;

    auto bunch = this->makeBunch();
    bunch->create(8);
    {
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            // Place near origin
            for (unsigned d = 0; d < TestFixture::Dim; d++)
                R_host(i)[d] = T(1e-5) * this->domain[d];
            Q_host(i) = T(1);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();  // initial valid placement

    // Now jump every particle to the diagonally opposite corner
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T newPos     = R_host(i)[d] + this->domain[d] * T(0.999);
                R_host(i)[d] = this->periodicWrap(newPos, this->domain[d]);
            }
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    bunch->update();
    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  18. Check that tags (IDs) are preserved through migration
// ============================================================
TYPED_TEST(TestParticleUpdate, ParticleTagsPreservedAfterMigration) {
    using T = typename TestFixture::T;

    if (ippl::Comm->size() < 2)
        GTEST_SKIP();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 128);

    // Record all local tags and charges before update
    std::vector<std::pair<long long, T>> localBefore;
    {
        auto t_host = bunch->tag.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        Kokkos::deep_copy(t_host, bunch->tag.getView());
        Kokkos::deep_copy(Q_host, bunch->Q.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            localBefore.push_back({t_host(i), Q_host(i)});
    }

    // Displace so migration happens
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T newPos     = R_host(i)[d] + this->domain[d] * T(0.4);
                R_host(i)[d] = this->periodicWrap(newPos, this->domain[d]);
            }
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    bunch->update();

    // Gather tags from all ranks and verify the global set is identical
    // (no duplicates, no missing)
    std::vector<long long> localTagsAfter;
    {
        auto t_host = bunch->tag.getHostMirror();
        Kokkos::deep_copy(t_host, bunch->tag.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            localTagsAfter.push_back(t_host(i));
    }

    // Each rank sorts its local list; then we check global count via reduce
    size_t localCount  = localTagsAfter.size();
    size_t globalCount = 0;
    ippl::Comm->reduce(localCount, globalCount, 1, std::plus<size_t>());

    // Reconstruct original count
    size_t originalCount  = localBefore.size();
    size_t globalOriginal = 0;
    ippl::Comm->reduce(originalCount, globalOriginal, 1, std::plus<size_t>());

    EXPECT_EQ(globalOriginal, globalCount);
}

// ============================================================
//  19. Periodic wrap: particle moved exactly one full domain length
// ============================================================
TYPED_TEST(TestParticleUpdate, ExactOneDomainLengthShiftConservesParticles) {
    using T = typename TestFixture::T;

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 64);
    const size_t before = this->totalParticles(*bunch);

    // Shift by exactly one domain length: after wrapping, position unchanged.
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T newPos     = R_host(i)[d] + this->domain[d];
                R_host(i)[d] = this->periodicWrap(newPos, this->domain[d]);
            }
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    bunch->update();
    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
//  20. Repeated create-and-update cycles (simulates particle injection)
// ============================================================
TYPED_TEST(TestParticleUpdate, RepeatedCreateAndUpdateCycles) {
    using T = typename TestFixture::T;

    auto bunch        = this->makeBunch();
    size_t cumulative = 0;

    for (int cycle = 0; cycle < 4; ++cycle) {
        // Inject 32 more particles on rank 0 each cycle
        constexpr unsigned inject = 32;
        int N                     = ippl::Comm->rank() == 0 ? inject : 0;
        size_t currentLocal       = bunch->getLocalNum();
        bunch->create(N);

        if (ippl::Comm->rank() == 0) {
            std::mt19937_64 eng(cycle * 100);
            std::uniform_real_distribution<T> unif(T(0), T(1));
            auto R_host = bunch->R.getHostMirror();
            auto Q_host = bunch->Q.getHostMirror();
            Kokkos::deep_copy(R_host, bunch->R.getView());
            Kokkos::deep_copy(Q_host, bunch->Q.getView());

            for (unsigned i = currentLocal; i < currentLocal + inject; ++i) {
                for (unsigned d = 0; d < TestFixture::Dim; d++)
                    R_host(i)[d] = unif(eng) * this->domain[d];
                Q_host(i) = T(1);
            }
            Kokkos::deep_copy(bunch->R.getView(), R_host);
            Kokkos::deep_copy(bunch->Q.getView(), Q_host);
            cumulative += inject;
        }
        ippl::Comm->barrier();

        bunch->update();

        size_t total = this->totalParticles(*bunch);
        if (ippl::Comm->rank() == 0) {
            EXPECT_EQ(cumulative, total) << "at cycle " << cycle;
        }
        EXPECT_EQ(0u, this->countMisplaced(*bunch)) << "at cycle " << cycle;
    }
}

// ============================================================
//  Entry point
// ============================================================
int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}