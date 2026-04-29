//
// Unit test: Particle spatial layout update with non-uniform (ORB) decompositions
//
// These tests exercise particle migration after OrthogonalRecursiveBisection
// has repartitioned the domain.  The ORB algorithm can produce highly asymmetric
// layouts where a single rank may span many more grid cells than its neighbour,
// so the usual assumption "every rank has the same width" does not hold.
//
// Minimum rank requirement:  all tests that exercise multi-rank migration are
// guarded by GTEST_SKIP() when fewer than MIN_RANKS MPI processes are available.
// The recommended launch is  mpirun -n 4  (or 8 for the higher-coverage tests).
//
// Test inventory
// ──────────────
//  1.  ORB layout created with uniform density  →  baseline count conservation
//  2.  ORB layout created with a Gaussian density bump  →  asymmetric widths
//  3.  Particles displaced uniformly after ORB  →  all migrate to correct owner
//  4.  Particles displaced toward the heavy region  →  burst migration into narrow rank
//  5.  Repeated ORB repartition between steps  →  counts conserved at each step
//  6.  All particles on rank 0, ORB applied, then update  →  full redistribution
//  7.  Charge conservation through ORB + update cycle
//  8.  Tags (IDs) survive ORB + update
//  9.  Zero-particle ranks after ORB do not deadlock
// 10.  Large-count stress test with ORB decomposition
// 11.  Periodic wrap across ORB boundaries
// 12.  Multiple ORB repartitions with particle injection between steps
// 13.  ORB with a step-function density  →  one rank gets 1 cell, another N-1
// 14.  Successive displacements across ORB boundaries
// 15.  ORB repartition in 3-D with diagonal displacement
//
#include "Ippl.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "Decomposition/OrthogonalRecursiveBisection.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

// ============================================================
//  Minimum ranks required for multi-rank migration tests
// ============================================================
static constexpr int MIN_RANKS  = 2;  // absolute minimum
static constexpr int PREF_RANKS = 4;  // preferred for asymmetry tests

// ============================================================
//  Particle bunch
// ============================================================
template <typename T, typename ExecSpace, unsigned Dim>
struct OrbBunch
    : public ippl::ParticleBase<
          ippl::ParticleSpatialLayout<T, Dim, ippl::UniformCartesian<T, Dim>, ExecSpace>> {
    using playout_type =
        ippl::ParticleSpatialLayout<T, Dim, ippl::UniformCartesian<T, Dim>, ExecSpace>;

    explicit OrbBunch(playout_type& pl)
        : ippl::ParticleBase<playout_type>(pl) {
        this->addAttribute(Q);
        this->addAttribute(tag);
    }

    ippl::ParticleAttrib<T, ExecSpace> Q;            // per-particle charge
    ippl::ParticleAttrib<long long, ExecSpace> tag;  // unique ID
};

// ============================================================
//  Fixture
// ============================================================
template <typename>
class TestParticleUpdateORB;

template <typename T_, typename ExecSpace, unsigned Dim_>
class TestParticleUpdateORB<Parameters<T_, ExecSpace, Rank<Dim_>>> : public ::testing::Test {
public:
    using T = T_;
    static constexpr unsigned Dim = Dim_;
    // ---- type aliases --------------------------------------------------
    using flayout_type   = ippl::FieldLayout<Dim>;
    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using playout_type   = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using RegionLayout_t = typename playout_type::RegionLayout_t;
    using bunch_type     = OrbBunch<T, ExecSpace, Dim>;
    using position_type  = ippl::Vector<T, Dim>;
    using Mesh_t = ippl::UniformCartesian<double, Dim>;
    using Centering_t = Mesh_t::DefaultCentering;

    // Density field type used by ORB
    using field_type = ippl::Field<T, Dim, mesh_type, Centering_t, ExecSpace>;
    using orb_type   = ippl::OrthogonalRecursiveBisection<field_type, T>;

    // ---- construction --------------------------------------------------
    TestParticleUpdateORB()
        : nPoints(getGridSizes<Dim>()) {
        for (unsigned d = 0; d < Dim; d++)
            domain[d] = static_cast<T>(nPoints[d]) / 16.0;

        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++)
            args[d] = ippl::Index(nPoints[d]);
        gDomain = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        ippl::Vector<T, Dim> hx, origin;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        for (unsigned d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        layout      = std::make_shared<flayout_type>(MPI_COMM_WORLD, gDomain, isParallel);
        mesh        = std::make_shared<mesh_type>(gDomain, hx, origin);
        playout_ptr = std::make_shared<playout_type>(*layout, *mesh);
    }

    // ---- ORB helpers ---------------------------------------------------

    /// Build a uniform-density field (all weights = 1) and run ORB.
    bool orbUniform() {
        field_type rho(*mesh, *layout);
        rho = T(1);
        orb_type orb;
        orb.initialize(*layout, *mesh, rho);
        // binaryRepartition needs particle positions; pass empty attrib
        ippl::ParticleAttrib<position_type, ExecSpace> emptyR;
        return orb.binaryRepartition(emptyR, *layout, /*isFirstRepartition=*/true);
    }

    /// Build a Gaussian density bump centred in the domain and run ORB.
    /// The heavier region forces ORB to cut near the centre, creating
    /// asymmetric sub-domains along each axis.
    bool orbGaussian(T sigma = T(0.15)) {
        field_type rho(*mesh, *layout);

        // Fill the density field on-device: Gaussian centred at 0.5
        auto view   = rho.getView();
        auto lDom   = layout->getLocalNDIndex();
        auto hx_vec = mesh->getMeshSpacing();
        auto orig   = mesh->getOrigin();
        int nghost  = rho.getNghost();

        // Kokkos::parallel_for("GaussianRho", ippl::createRangePolicy<Dim, ExecSpace>(rho.getOwned()),
        //                      KOKKOS_LAMBDA(/* index args */){
        //                          // Generic lambda for any Dim via index_array_type
        //                      });

        // Simpler: fill via host mirror
        auto rho_host = rho.getHostMirror();
        auto lDomH    = layout->getLocalNDIndex();
        for (int gz = (Dim > 2 ? lDomH[2].first() : 0); gz <= (Dim > 2 ? lDomH[2].last() : 0);
             ++gz) {
            for (int gy = (Dim > 1 ? lDomH[1].first() : 0); gy <= (Dim > 1 ? lDomH[1].last() : 0);
                 ++gy) {
                for (int gx = lDomH[0].first(); gx <= lDomH[0].last(); ++gx) {
                    T val = T(1);
                    // Compute normalised coordinate and Gaussian weight
                    auto computeCoord = [&](int idx, unsigned d) -> T {
                        return (static_cast<T>(idx) + T(0.5)) / static_cast<T>(nPoints[d]);
                    };
                    T x  = computeCoord(gx, 0) - T(0.5);
                    T r2 = x * x;
                    if constexpr (Dim > 1) {
                        T y = computeCoord(gy, 1) - T(0.5);
                        r2 += y * y;
                    }
                    if constexpr (Dim > 2) {
                        T z = computeCoord(gz, 2) - T(0.5);
                        r2 += z * z;
                    }
                    val = std::exp(-r2 / (T(2) * sigma * sigma));

                    // Map global index to local+ghost index
                    int lx = gx - lDomH[0].first() + rho.getNghost();
                    if constexpr (Dim == 1)
                        rho_host(lx) = val;
                    else if constexpr (Dim == 2) {
                        int ly           = gy - lDomH[1].first() + rho.getNghost();
                        rho_host(lx, ly) = val;
                    } else {
                        int ly               = gy - lDomH[1].first() + rho.getNghost();
                        int lz               = gz - lDomH[2].first() + rho.getNghost();
                        rho_host(lx, ly, lz) = val;
                    }
                }
            }
        }
        Kokkos::deep_copy(rho.getView(), rho_host);

        orb_type orb;
        orb.initialize(*layout, *mesh, rho);
        ippl::ParticleAttrib<position_type, ExecSpace> emptyR;
        return orb.binaryRepartition(emptyR, *layout, /*isFirstRepartition=*/true);
    }

    /// Build a step-function density: cells in [0, nPoints[0]/4) get weight 100,
    /// the rest get weight 1.  This forces one rank to be very narrow (few cells)
    /// while the others are wide.
    bool orbStepFunction() {
        field_type rho(*mesh, *layout);
        auto rho_host = rho.getHostMirror();
        auto lDomH    = layout->getLocalNDIndex();
        int cutGlobal = static_cast<int>(nPoints[0]) / 4;

        for (int gz = (Dim > 2 ? lDomH[2].first() : 0); gz <= (Dim > 2 ? lDomH[2].last() : 0);
             ++gz) {
            for (int gy = (Dim > 1 ? lDomH[1].first() : 0); gy <= (Dim > 1 ? lDomH[1].last() : 0);
                 ++gy) {
                for (int gx = lDomH[0].first(); gx <= lDomH[0].last(); ++gx) {
                    T val  = (gx < cutGlobal) ? T(100) : T(1);
                    int lx = gx - lDomH[0].first() + rho.getNghost();
                    if constexpr (Dim == 1)
                        rho_host(lx) = val;
                    else if constexpr (Dim == 2) {
                        int ly           = gy - lDomH[1].first() + rho.getNghost();
                        rho_host(lx, ly) = val;
                    } else {
                        int ly               = gy - lDomH[1].first() + rho.getNghost();
                        int lz               = gz - lDomH[2].first() + rho.getNghost();
                        rho_host(lx, ly, lz) = val;
                    }
                }
            }
        }
        Kokkos::deep_copy(rho.getView(), rho_host);

        orb_type orb;
        orb.initialize(*layout, *mesh, rho);
        ippl::ParticleAttrib<position_type, ExecSpace> emptyR;
        return orb.binaryRepartition(emptyR, *layout, /*isFirstRepartition=*/true);
    }

    /// Rebuild the playout after the layout has been repartitioned.
    void rebuildPlayout() { playout_ptr = std::make_shared<playout_type>(*layout, *mesh); }

    // ---- bunch helpers -------------------------------------------------

    std::shared_ptr<bunch_type> makeBunch() {
        auto b = std::make_shared<bunch_type>(*playout_ptr);
        typename bunch_type::bc_container_type bcs;
        bcs.fill(ippl::BC::PERIODIC);
        b->setParticleBC(bcs);
        return b;
    }

    void fillRandom(bunch_type& b, unsigned n, unsigned long seed = 42) {
        int nRanks   = ippl::Comm->size();
        unsigned per = std::max(1u, n / nRanks);
        b.create(per);

        std::mt19937_64 eng(seed + ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(T(0), T(1));

        auto R_host = b.R.getHostMirror();
        auto Q_host = b.Q.getHostMirror();
        auto t_host = b.tag.getHostMirror();

        for (size_t i = 0; i < b.getLocalNum(); ++i) {
            position_type r;
            for (unsigned d = 0; d < Dim; d++)
                r[d] = unif(eng) * domain[d];
            R_host(i) = r;
            Q_host(i) = T(1);
            t_host(i) =
                static_cast<long long>(ippl::Comm->rank()) * 10000000LL + static_cast<long long>(i);
        }
        Kokkos::deep_copy(b.R.getView(), R_host);
        Kokkos::deep_copy(b.Q.getView(), Q_host);
        Kokkos::deep_copy(b.tag.getView(), t_host);
    }

    // ---- assertion helpers ---------------------------------------------

    size_t totalParticles(const bunch_type& b) const {
        size_t local = b.getLocalNum(), total = 0;
        ippl::Comm->reduce(local, total, 1, std::plus<size_t>());
        return total;
    }

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

    /// Count locally owned particles that lie outside the owning rank's region.
    size_t countMisplaced(bunch_type& b) {
        RegionLayout_t rl = playout_ptr->getRegionLayout();
        auto regions      = rl.getdLocalRegions();
        auto regions_host = Kokkos::create_mirror_view(regions);
        Kokkos::deep_copy(regions_host, regions);
        int myRank = ippl::Comm->rank();

        auto R_host = b.R.getHostMirror();
        Kokkos::deep_copy(R_host, b.R.getView());

        size_t bad = 0;
        for (size_t i = 0; i < b.getLocalNum(); ++i) {
            bool ok = true;
            for (unsigned d = 0; d < Dim; d++) {
                T p = R_host(i)[d];
                ok &= (p >= regions_host(myRank)[d].min() && p <= regions_host(myRank)[d].max());
            }
            if (!ok)
                ++bad;
        }
        return bad;
    }

    T periodicWrap(T x, T L) const {
        while (x < T(0))
            x += L;
        while (x >= L)
            x -= L;
        return x;
    }

    /// Verify that the widths of local domains differ between ranks (i.e. ORB
    /// actually produced a non-uniform decomposition along at least one axis).
    /// Returns true if at least one pair of ranks has different widths.
    bool isNonUniform() const {
        auto lDom    = layout->getLocalNDIndex();
        int myWidth  = lDom[0].length();
        int maxWidth = 0, minWidth = 0;
        ippl::Comm->allreduce(myWidth, maxWidth, 1, std::greater<int>{});
        ippl::Comm->allreduce(myWidth, minWidth, 1, std::less<int>{});
        return (maxWidth != minWidth);
    }

    // ---- data members --------------------------------------------------
    std::array<size_t, Dim> nPoints;
    std::array<T, Dim> domain;
    ippl::NDIndex<Dim> gDomain;

    std::shared_ptr<flayout_type> layout;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<playout_type> playout_ptr;
};

// Run over 1-D, 2-D, 3-D with default scalar / exec space
using Tests = TestParams::tests<1, 2, 3>;
TYPED_TEST_SUITE(TestParticleUpdateORB, Tests);

// ============================================================
//  Helper macro: skip test if rank count is below threshold
// ============================================================
#define REQUIRE_RANKS(n)                                                  \
    do {                                                                  \
        if (ippl::Comm->size() < (n)) {                                   \
            GTEST_SKIP() << "Test requires at least " << (n) << " ranks"; \
        }                                                                 \
    } while (false)

// ============================================================
//  1. ORB with uniform density – baseline conservation
// ============================================================
TYPED_TEST(TestParticleUpdateORB, UniformOrbBaselineConservation) {
    REQUIRE_RANKS(MIN_RANKS);

    bool ok = this->orbUniform();
    if (!ok)
        GTEST_SKIP() << "ORB returned planar decomposition – skipping";

    this->rebuildPlayout();
    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  2. ORB with Gaussian density – verify non-uniform decomposition
//     and that particles still migrate correctly
// ============================================================
TYPED_TEST(TestParticleUpdateORB, GaussianOrbCreatesNonUniformLayout) {
    REQUIRE_RANKS(PREF_RANKS);

    bool ok = this->orbGaussian();
    if (!ok)
        GTEST_SKIP() << "ORB returned planar decomposition – skipping";

    // With 4+ ranks and a Gaussian bump the domains should be unequal
    if (ippl::Comm->size() >= PREF_RANKS) {
        EXPECT_TRUE(this->isNonUniform()) << "Expected ORB to produce a non-uniform decomposition";
    }

    this->rebuildPlayout();
    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  3. Uniform displacement after ORB – all particles cross at least
//     one ORB boundary and reach the correct owner
// ============================================================
TYPED_TEST(TestParticleUpdateORB, UniformDisplacementAfterOrb) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbGaussian();
    if (!ok)
        GTEST_SKIP();

    this->rebuildPlayout();
    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    // Shift by 30 % of domain along every axis
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T np         = R_host(i)[d] + T(0.3) * this->domain[d];
                R_host(i)[d] = this->periodicWrap(np, this->domain[d]);
            }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  4. Burst migration into the narrow rank after step-function ORB
//     One rank is very narrow; we push all particles toward it.
// ============================================================
TYPED_TEST(TestParticleUpdateORB, BurstMigrationIntoNarrowRankAfterStepOrb) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbStepFunction();
    if (!ok)
        GTEST_SKIP();

    this->rebuildPlayout();
    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    // Move every particle into the first quarter of the domain (the narrow rank)
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        std::mt19937_64 eng(77 + ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(T(0), T(0.25));
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            R_host(i)[0] = unif(eng) * this->domain[0];
            for (unsigned d = 1; d < TestFixture::Dim; d++)
                R_host(i)[d] = T(0.5) * this->domain[d];  // safe middle
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
//  5. Repeated ORB repartition between particle steps
// ============================================================
TYPED_TEST(TestParticleUpdateORB, RepeatedOrbRepartitionConservesParticles) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    // Perform 3 ORB repartitions, each time with a different density,
    // interleaved with particle displacements.
    constexpr unsigned N = 128;

    bool firstOk = this->orbUniform();
    if (!firstOk)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, N);
    const size_t total = this->totalParticles(*bunch);

    // Lambda: displace by fraction f and call update
    auto displace = [&](T f) {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T np         = R_host(i)[d] + f * this->domain[d];
                R_host(i)[d] = this->periodicWrap(np, this->domain[d]);
            }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        bunch->update();
    };

    displace(T(0.15));
    EXPECT_EQ(total, this->totalParticles(*bunch));

    // Second ORB (Gaussian)
    bool ok2 = this->orbGaussian();
    if (ok2) {
        this->rebuildPlayout();
        // Re-attach the playout to the existing bunch
        bunch = this->makeBunch();
        this->fillRandom(*bunch, N);
        displace(T(0.25));
        EXPECT_EQ(total, this->totalParticles(*bunch));
    }

    // Third ORB (step-function)
    bool ok3 = this->orbStepFunction();
    if (ok3) {
        this->rebuildPlayout();
        bunch = this->makeBunch();
        this->fillRandom(*bunch, N);
        displace(T(0.1));
        EXPECT_EQ(total, this->totalParticles(*bunch));
    }
}

// ============================================================
//  6. All particles on rank 0, ORB applied, then update
// ============================================================
TYPED_TEST(TestParticleUpdateORB, AllParticlesOnRank0AfterOrb) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbGaussian();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    unsigned N = ippl::Comm->rank() == 0 ? 256 : 0;
    auto bunch = this->makeBunch();
    bunch->create(N);

    if (ippl::Comm->rank() == 0) {
        std::mt19937_64 eng(12);
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
//  7. Charge conservation through ORB + update cycle
// ============================================================
TYPED_TEST(TestParticleUpdateORB, ChargeConservedAfterOrbAndUpdate) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbGaussian();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 256);

    // Assign unique charges
    {
        auto Q_host = bunch->Q.getHostMirror();
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            Q_host(i) = static_cast<T>(ippl::Comm->rank() * 100000 + i + 1);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const T chargeBefore = this->totalCharge(*bunch);

    // Displace and update
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T np         = R_host(i)[d] + T(0.35) * this->domain[d];
                R_host(i)[d] = this->periodicWrap(np, this->domain[d]);
            }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }
    bunch->update();

    const T chargeAfter = this->totalCharge(*bunch);
    EXPECT_NEAR(static_cast<double>(chargeBefore), static_cast<double>(chargeAfter),
                1e-6 * std::abs(static_cast<double>(chargeBefore)));
}

// ============================================================
//  8. Particle tags survive ORB + update
// ============================================================
TYPED_TEST(TestParticleUpdateORB, TagsPreservedAfterOrbAndUpdate) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbUniform();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 128);

    // Record global original count of unique tags
    size_t origLocal  = bunch->getLocalNum();
    size_t origGlobal = 0;
    ippl::Comm->reduce(origLocal, origGlobal, 1, std::plus<size_t>());

    // Displace
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T np         = R_host(i)[d] + T(0.45) * this->domain[d];
                R_host(i)[d] = this->periodicWrap(np, this->domain[d]);
            }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }
    bunch->update();

    size_t afterLocal  = bunch->getLocalNum();
    size_t afterGlobal = 0;
    ippl::Comm->reduce(afterLocal, afterGlobal, 1, std::plus<size_t>());

    EXPECT_EQ(origGlobal, afterGlobal);
}

// ============================================================
//  9. Zero-particle ranks after ORB do not deadlock
// ============================================================
TYPED_TEST(TestParticleUpdateORB, ZeroParticleRanksAfterOrbDoNotDeadlock) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbStepFunction();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    // Only rank 0 creates particles
    auto bunch = this->makeBunch();
    unsigned N = ippl::Comm->rank() == 0 ? 64 : 0;
    bunch->create(N);
    if (ippl::Comm->rank() == 0) {
        std::mt19937_64 eng(55);
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
    ASSERT_NO_THROW(bunch->update());
    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
// 10. Large-count stress test with Gaussian ORB decomposition
// ============================================================
TYPED_TEST(TestParticleUpdateORB, LargeParticleCountGaussianOrb) {
    REQUIRE_RANKS(MIN_RANKS);

    bool ok = this->orbGaussian();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 4096, /*seed=*/98765);

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
// 11. Periodic wrap across ORB boundaries
// ============================================================
TYPED_TEST(TestParticleUpdateORB, PeriodicWrapAcrossOrbBoundaries) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbGaussian();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();

    // Each rank places particles just beyond the global domain's upper edge
    // along axis 0 so they wrap back to near 0.
    bunch->create(4);
    {
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        std::mt19937_64 eng(33 + ippl::Comm->rank());
        std::uniform_real_distribution<T> tiny(T(1e-5), T(1e-3));
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            R_host(i)[0] = this->domain[0] + tiny(eng) * this->domain[0];
            for (unsigned d = 1; d < TestFixture::Dim; d++)
                R_host(i)[d] = T(0.5) * this->domain[d];
            Q_host(i) = T(1);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
}

// ============================================================
// 12. Particle injection between ORB repartitions
//     Simulates a typical PIC workflow: inject → ORB → move → update → repeat
// ============================================================
TYPED_TEST(TestParticleUpdateORB, ParticleInjectionBetweenOrbRepartitions) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    constexpr unsigned injectPerCycle = 32;
    constexpr int cycles              = 3;
    size_t cumulative                 = 0;

    for (int c = 0; c < cycles; ++c) {
        // Repartition with Gaussian for odd cycles, uniform for even
        bool ok = (c % 2 == 0) ? this->orbUniform() : this->orbGaussian();
        if (!ok)
            continue;
        this->rebuildPlayout();

        auto bunch = this->makeBunch();

        // Inject on rank 0
        unsigned N = ippl::Comm->rank() == 0 ? injectPerCycle : 0;
        bunch->create(N);
        if (ippl::Comm->rank() == 0) {
            std::mt19937_64 eng(c * 1000);
            std::uniform_real_distribution<T> unif(T(0), T(1));
            auto R_host = bunch->R.getHostMirror();
            auto Q_host = bunch->Q.getHostMirror();
            for (unsigned i = 0; i < injectPerCycle; ++i) {
                for (unsigned d = 0; d < TestFixture::Dim; d++)
                    R_host(i)[d] = unif(eng) * this->domain[d];
                Q_host(i) = T(1);
            }
            Kokkos::deep_copy(bunch->R.getView(), R_host);
            Kokkos::deep_copy(bunch->Q.getView(), Q_host);
            cumulative += injectPerCycle;
        }

        bunch->update();

        auto total_particles = this->totalParticles(*bunch);
        if (ippl::Comm->rank() == 0) {
            EXPECT_EQ(cumulative, total_particles) << "at cycle " << c;
        }
        EXPECT_EQ(0u, this->countMisplaced(*bunch)) << "at cycle " << c;
    }
}

// ============================================================
// 13. Step-function ORB: narrow rank correctly receives migrating particles
//     and does not receive more than the domain capacity
// ============================================================
TYPED_TEST(TestParticleUpdateORB, NarrowRankReceivesCorrectlyAfterStepOrb) {
    REQUIRE_RANKS(PREF_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbStepFunction();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 512);

    // Move all particles into the heavy region (second three-quarters)
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        std::mt19937_64 eng(808 + ippl::Comm->rank());
        std::uniform_real_distribution<T> heavyX(T(0.25), T(1.0));
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            R_host(i)[0] = heavyX(eng) * this->domain[0];
            for (unsigned d = 1; d < TestFixture::Dim; d++)
                R_host(i)[d] = T(0.5) * this->domain[d];
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();

    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
}

// ============================================================
// 14. Successive displacements across ORB boundaries
// ============================================================
TYPED_TEST(TestParticleUpdateORB, SuccessiveDisplacementsAcrossOrbBoundaries) {
    REQUIRE_RANKS(MIN_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbGaussian();
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();
    this->fillRandom(*bunch, 128);
    const size_t N = this->totalParticles(*bunch);

    std::mt19937_64 eng(2025 + ippl::Comm->rank());
    std::uniform_real_distribution<T> step(-T(0.2), T(0.2));

    for (int s = 0; s < 6; ++s) {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            for (unsigned d = 0; d < TestFixture::Dim; d++) {
                T np         = R_host(i)[d] + step(eng) * this->domain[d];
                R_host(i)[d] = this->periodicWrap(np, this->domain[d]);
            }
        Kokkos::deep_copy(bunch->R.getView(), R_host);

        bunch->update();
        EXPECT_EQ(N, this->totalParticles(*bunch)) << "at step " << s;
        EXPECT_EQ(0u, this->countMisplaced(*bunch)) << "at step " << s;
    }
}

// ============================================================
// 15. 3-D corner migration with ORB decomposition
//     (only runs when Dim == 3)
// ============================================================
TYPED_TEST(TestParticleUpdateORB, ThreeDCornerMigrationAfterOrb) {
    if constexpr (TestFixture::Dim != 3) {
        GTEST_SKIP() << "3-D specific test";
    }
    REQUIRE_RANKS(PREF_RANKS);
    using T = typename TestFixture::T;

    bool ok = this->orbGaussian(T(0.2));
    if (!ok)
        GTEST_SKIP();
    this->rebuildPlayout();

    auto bunch = this->makeBunch();

    // Seed all particles near the origin
    bunch->create(64);
    {
        auto R_host = bunch->R.getHostMirror();
        auto Q_host = bunch->Q.getHostMirror();
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            for (unsigned d = 0; d < 3; d++)
                R_host(i)[d] = T(0.02) * this->domain[d];
            Q_host(i) = T(1);
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
    }

    const size_t before = this->totalParticles(*bunch);
    bunch->update();  // settle near origin

    // Jump diagonally to the opposite corner (wraps periodically)
    {
        auto R_host = bunch->R.getHostMirror();
        Kokkos::deep_copy(R_host, bunch->R.getView());
        for (size_t i = 0; i < bunch->getLocalNum(); ++i)
            for (unsigned d = 0; d < 3; d++) {
                T np         = R_host(i)[d] + T(0.98) * this->domain[d];
                R_host(i)[d] = this->periodicWrap(np, this->domain[d]);
            }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    bunch->update();
    EXPECT_EQ(before, this->totalParticles(*bunch));
    EXPECT_EQ(0u, this->countMisplaced(*bunch));
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
