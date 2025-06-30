//
// Unit test Particle send/receive
//   Test particle send and receive operations.
//
//
#include "Ippl.h"

#include <random>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class ParticleSendRecv;

template <typename T, typename ExecSpace, unsigned Dim>
class ParticleSendRecv<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using flayout_type   = ippl::FieldLayout<Dim>;
    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using playout_type   = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using RegionLayout_t = typename playout_type::RegionLayout_t;

    using rank_type = ippl::ParticleAttrib<int, ExecSpace>;

    template <class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        explicit Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(expectedRank);
            this->addAttribute(Q);
        }

        ~Bunch() = default;

        using charge_container_type = ippl::ParticleAttrib<T>;

        rank_type expectedRank;
        charge_container_type Q;
    };

    using bunch_type = Bunch<playout_type>;

    ParticleSendRecv()
        : nPoints(getGridSizes<Dim>()) {
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = nPoints[d] / 16.;
        }

        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++) {
            args[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        for (unsigned int d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        layout  = flayout_type(MPI_COMM_WORLD, owned, isParallel);
        mesh    = mesh_type(owned, hx, origin);
        playout_ptr = std::make_shared<playout_type>(layout, mesh);
        bunch = std::make_shared<bunch_type>(*playout_ptr);

        using BC = ippl::BC;

        typename bunch_type::bc_container_type bcs;
        bcs.fill(BC::PERIODIC);

        bunch->setParticleBC(bcs);

        int nRanks = ippl::Comm->size();
        if (nParticles % nRanks > 0) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
        }

        bunch->create(nParticles / nRanks);

        std::mt19937_64 eng(ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(0, 1);

        auto R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            ippl::Vector<T, Dim> r;
            for (unsigned d = 0; d < Dim; d++) {
                r[d] = unif(eng) * domain[d];
            }
            R_host(i) = r;
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
        bunch->Q = 1.0;

        computeExpectedRanks();
    }

    void computeExpectedRanks() {
        using region_view  = typename RegionLayout_t::view_type;
        using size_type    = typename RegionLayout_t::view_type::size_type;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;

        RegionLayout_t RLayout           = playout_ptr->getRegionLayout();
        auto& positions                  = bunch->R.getView();
        region_view Regions              = RLayout.getdLocalRegions();
        typename rank_type::view_type ER = bunch->expectedRank.getView();

        Kokkos::parallel_for(
            "Expected Rank", mdrange_type({0, 0}, {ER.extent(0), Regions.extent(0)}),
            KOKKOS_LAMBDA(const size_t i, const size_type j) {
                bool xyz_bool = true;
                for (unsigned d = 0; d < Dim; d++) {
                    xyz_bool &= positions(i)[d] <= Regions(j)[d].max()
                                && positions(i)[d] >= Regions(j)[d].min();
                }
                if (xyz_bool) {
                    ER(i) = j;
                }
            });
        Kokkos::fence();
    }

    std::shared_ptr<bunch_type> bunch;
    const unsigned int nParticles = 128;
    std::array<size_t, Dim> nPoints;
    std::array<T, Dim> domain;
    std::shared_ptr<playout_type> playout_ptr;

    flayout_type layout;
    mesh_type mesh;
};

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_SUITE(ParticleSendRecv, Tests);

TYPED_TEST(ParticleSendRecv, SendAndRecieve) {
    const auto nParticles = this->nParticles;
    auto& bunch           = this->bunch;

    bunch->update();
    typename TestFixture::rank_type::view_type::host_mirror_type ER_host =
        bunch->expectedRank.getHostMirror();

    Kokkos::resize(ER_host, bunch->expectedRank.size());
    Kokkos::deep_copy(ER_host, bunch->expectedRank.getView());

    for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
        ASSERT_EQ(ER_host(i), ippl::Comm->rank());
    }
    ippl::Comm->barrier();

    unsigned int Total_particles = 0;
    unsigned int local_particles = bunch->getLocalNum();

    ippl::Comm->reduce(local_particles, Total_particles, 1, std::plus<unsigned int>());

    if (ippl::Comm->rank() == 0) {
        ASSERT_EQ(nParticles, Total_particles);
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
