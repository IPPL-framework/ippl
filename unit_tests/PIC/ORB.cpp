//
// Unit tests ORB for class OrthogonalRecursiveBisection
//   Test volume and charge conservation in PIC operations.
//
//
#include "Ippl.h"

#include <random>
#include <vector>

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

TEST(ORBRegressionTest, EnforcesMinimumChildWidth) {
    if (ippl::Comm->size() != 2) {
        GTEST_SKIP() << "This regression exercises a single two-rank split.";
    }

    using Mesh   = ippl::UniformCartesian<double, 3>;
    using Field  = ippl::Field<double, 3, Mesh, Mesh::DefaultCentering>;
    using Layout = ippl::FieldLayout<3>;
    using Box    = ippl::NDIndex<3>;

    for (int extent : {2, 3, 4}) {
        SCOPED_TRACE(extent);
        Box domain{ippl::Index(extent), ippl::Index(extent), ippl::Index(extent)};
        Layout layout(MPI_COMM_WORLD, domain, {true, true, true});
        Mesh mesh(domain, ippl::Vector<double, 3>(1.0), ippl::Vector<double, 3>(0.0));
        Field weights(mesh, layout);
        weights = 0.0;

        std::vector<Box> originalDomains;
        for (int rank = 0; rank < ippl::Comm->size(); ++rank) {
            originalDomains.push_back(layout.getLocalNDIndex(rank));
        }
        ippl::OrthogonalRecursiveBisection<Field> orb;
        orb.initialize(layout, mesh, weights);
        ippl::ParticleAttrib<ippl::Vector<double, 3>> unusedPositions;

        const bool success = orb.binaryRepartition(unusedPositions, layout, true);
        EXPECT_EQ(success, extent == 4);
        for (int rank = 0; rank < ippl::Comm->size(); ++rank) {
            const auto& current = layout.getLocalNDIndex(rank);
            for (unsigned d = 0; d < 3; ++d) {
                if (extent < 4) {
                    EXPECT_EQ(current[d], originalDomains[rank][d]);
                } else {
                    EXPECT_GE(current[d].length(), 2u);
                }
            }
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
