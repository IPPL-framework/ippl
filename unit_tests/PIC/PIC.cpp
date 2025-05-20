//
// Unit test PICTest
//   Test scatter and gather particle-in-cell operations.
//
#include "Ippl.h"

#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class PICTest;

template <typename T, typename ExecSpace, unsigned Dim>
class PICTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    constexpr static unsigned dim = Dim;
    using value_type              = T;

    using mesh_type      = ippl::UniformCartesian<double, Dim>;
    using centering_type = typename mesh_type::DefaultCentering;
    using field_type     = ippl::Field<double, Dim, mesh_type, centering_type, ExecSpace>;
    using flayout_type   = ippl::FieldLayout<Dim>;
    using playout_type   = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;

    template <class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        explicit Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(Q);
        }

        ~Bunch() = default;

        ippl::ParticleAttrib<double, ExecSpace> Q;
    };

    using bunch_type = Bunch<playout_type>;

    PICTest()
        : nPoints(getGridSizes<Dim>()) {
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = nPoints[d] / 16.;
        }

        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++) {
            args[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        std::array<bool, Dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        isParallel.fill(true);

        for (unsigned int d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        layout = flayout_type(MPI_COMM_WORLD, owned, isParallel);
        mesh   = mesh_type(owned, hx, origin);

        field = std::make_unique<field_type>(mesh, layout);

        playout_ptr = std::make_shared<playout_type>(layout,mesh);

        bunch = std::make_shared<bunch_type>(*playout_ptr);

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

        auto R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < nloc; ++i) {
            for (unsigned d = 0; d < Dim; d++) {
                std::uniform_real_distribution<double> unif(hx[0] / 2, domain[d] - (hx[0] / 2));
                R_host(i)[d] = unif(eng);
            }
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    std::shared_ptr<field_type> field;
    std::shared_ptr<bunch_type> bunch;
    size_t nParticles = 32;
    std::array<size_t, Dim> nPoints;
    std::array<double, Dim> domain;

    std::shared_ptr<playout_type> playout_ptr;
    flayout_type layout;
    mesh_type mesh;
};

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_SUITE(PICTest, Tests);

TYPED_TEST(PICTest, Scatter) {
    auto& field      = this->field;
    auto& bunch      = this->bunch;
    auto& nParticles = this->nParticles;

    *field = 0.0;

    double charge = 0.5;

    bunch->Q = charge;

    bunch->update();

    scatter(bunch->Q, *field, bunch->R);

    double totalcharge = field->sum();
    totalcharge = 0.0;  // this should trigger an error in the CI/CD framework
    ASSERT_NEAR((nParticles * charge - totalcharge) / (nParticles * charge), 0.0,
                tolerance<typename TestFixture::value_type>);
}

TYPED_TEST(PICTest, Gather) {
    auto& field      = this->field;
    auto& bunch      = this->bunch;
    auto& nParticles = this->nParticles;

    *field = 1.0;

    bunch->Q = 0.0;

    bunch->update();

    gather(bunch->Q, *field, bunch->R);

    ASSERT_DOUBLE_EQ((nParticles - bunch->Q.sum()) / nParticles, 0.0);
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
