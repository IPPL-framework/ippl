//
// Unit tests ORB for class OrthogonalRecursiveBisection
//   Test volume and charge conservation in PIC operations.
//
//
#include "Ippl.h"

#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class ORBTest;

template <typename T, typename ExecSpace, unsigned Dim>
class ORBTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    constexpr static unsigned dim = Dim;
    using value_type              = T;

    using mesh_type      = ippl::UniformCartesian<double, Dim>;
    using centering_type = typename mesh_type::DefaultCentering;
    using field_type     = ippl::Field<double, Dim, mesh_type, centering_type, ExecSpace>;
    using flayout_type   = ippl::FieldLayout<Dim>;
    using playout_type   = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using ORB            = ippl::OrthogonalRecursiveBisection<field_type>;

    template <class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        explicit Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(Q);
        }

        ~Bunch() = default;

        ippl::ParticleAttrib<double, ExecSpace> Q;

        void updateLayout(flayout_type fl, mesh_type mesh) {
            PLayout& layout = this->getLayout();
            layout.updateLayout(fl, mesh);
        }
    };

    using bunch_type = Bunch<playout_type>;

    ORBTest()
        : nPoints(getGridSizes<Dim>()) {
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = nPoints[d] / 32.;
        }

        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++)
            args[d] = ippl::Index(nPoints[d]);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        std::array<bool, Dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        for (unsigned int d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        const bool isAllPeriodic = true;
        layout                   = flayout_type(MPI_COMM_WORLD, owned, isParallel, isAllPeriodic);
        mesh                     = mesh_type(owned, hx, origin);
        field                    = std::make_shared<field_type>(mesh, layout);
        playout_ptr              = std::make_shared<playout_type>(layout, mesh);
        bunch                    = std::make_shared<bunch_type>(*playout_ptr);

        int nRanks = ippl::Comm->size();
        if (nParticles % nRanks > 0) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
            exit(1);
        }

        size_t nloc = nParticles / nRanks;
        bunch->create(nloc);

        std::mt19937_64 eng;
        eng.seed(42);
        eng.discard(nloc * ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(0, 1);

        auto R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < nloc; ++i) {
            for (unsigned d = 0; d < Dim; d++) {
                R_host(i)[d] = unif(eng) * domain[d];
            }
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);

        orb.initialize(layout, mesh, *field);
    }

    void repartition() {
        bool fromAnalyticDensity = false;

        orb.binaryRepartition(bunch->R, layout, fromAnalyticDensity);
        field->updateLayout(layout);
        bunch->updateLayout(layout, mesh);
    }

    std::shared_ptr<field_type> field;
    std::shared_ptr<bunch_type> bunch;
    size_t nParticles = 128;
    std::array<size_t, Dim> nPoints;
    std::array<double, Dim> domain;

    flayout_type layout;
    mesh_type mesh;
    std::shared_ptr<playout_type> playout_ptr;
    ORB orb;
};

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_SUITE(ORBTest, Tests);

TYPED_TEST(ORBTest, Volume) {
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch  = this->bunch;
    auto& layout = this->layout;

    ippl::NDIndex<Dim> dom = layout.getDomain();

    bunch->update();

    this->repartition();

    bunch->update();

    ippl::NDIndex<Dim> ndom = layout.getDomain();

    ASSERT_DOUBLE_EQ(dom.size(), ndom.size());
}

TYPED_TEST(ORBTest, Charge) {
    auto& bunch = this->bunch;
    auto& field = this->field;

    typename TestFixture::value_type tol = tolerance<typename TestFixture::value_type>;

    double charge = 0.5;
    bunch->Q = charge/this->nParticles;

    bunch->update();

    this->repartition();

    bunch->update();

    *field = 0.0;
    scatter(bunch->Q, *field, bunch->R);
    double totalCharge = field->sum();

    ASSERT_NEAR((charge - totalCharge), 0., tol);

}

TYPED_TEST(ORBTest, AllowedAxesPreserveDisabledDimensions) {
    constexpr unsigned Dim = TestFixture::dim;

    if constexpr (Dim < 2) {
        GTEST_SKIP() << "Need at least one disabled dimension.";
    } else {
        if (ippl::Comm->size() < 2) {
            GTEST_SKIP() << "Allowed-axis repartition needs more than one rank.";
        }

        auto& bunch  = this->bunch;
        auto& layout = this->layout;
        auto& orb    = this->orb;

        std::array<bool, Dim> allowedAxes;
        allowedAxes.fill(false);
        allowedAxes[0] = true;

        bool fromAnalyticDensity = true;
        ASSERT_TRUE(orb.binaryRepartition(bunch->R, layout, fromAnalyticDensity, allowedAxes));

        const auto& globalDomain = layout.getDomain();
        auto localDomains        = layout.getHostLocalDomains();

        bool cutAllowedAxis = false;
        for (size_t rank = 0; rank < localDomains.size(); ++rank) {
            const auto& localDomain = localDomains(rank);
            if (localDomain[0].first() != globalDomain[0].first()
                || localDomain[0].last() != globalDomain[0].last()) {
                cutAllowedAxis = true;
            }

            for (unsigned d = 1; d < Dim; ++d) {
                EXPECT_EQ(localDomain[d].first(), globalDomain[d].first());
                EXPECT_EQ(localDomain[d].last(), globalDomain[d].last());
                EXPECT_EQ(localDomain[d].stride(), globalDomain[d].stride());
            }
        }

        EXPECT_TRUE(cutAllowedAxis);
    }
}

TYPED_TEST(ORBTest, RejectsEmptyAllowedAxisMask) {
    constexpr unsigned Dim = TestFixture::dim;

    if (ippl::Comm->size() < 2) {
        GTEST_SKIP() << "Empty allowed-axis mask only fails when ORB needs to cut.";
    }

    auto& bunch  = this->bunch;
    auto& layout = this->layout;
    auto& orb    = this->orb;

    auto originalDomains = layout.getHostLocalDomains();

    std::array<bool, Dim> allowedAxes;
    allowedAxes.fill(false);

    bool fromAnalyticDensity = true;
    EXPECT_FALSE(orb.binaryRepartition(bunch->R, layout, fromAnalyticDensity, allowedAxes));

    auto currentDomains = layout.getHostLocalDomains();
    ASSERT_EQ(originalDomains.size(), currentDomains.size());

    for (size_t rank = 0; rank < currentDomains.size(); ++rank) {
        for (unsigned d = 0; d < Dim; ++d) {
            EXPECT_EQ(currentDomains(rank)[d].first(), originalDomains(rank)[d].first());
            EXPECT_EQ(currentDomains(rank)[d].last(), originalDomains(rank)[d].last());
            EXPECT_EQ(currentDomains(rank)[d].stride(), originalDomains(rank)[d].stride());
        }
    }
}

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
