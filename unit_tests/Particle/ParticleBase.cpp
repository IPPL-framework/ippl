#include "Ippl.h"

#include <cmath>
#include "gtest/gtest.h"

class ParticleBaseTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::detail::ParticleLayout<double, dim> playout;
    typedef ippl::ParticleBase<playout> bunch_type;

    ParticleBaseTest() {
        setup();
    }

    void setup() {
        std::shared_ptr<playout> pl = std::make_shared<playout>();
        pbase = std::make_unique<bunch_type>(pl);
    }

    std::unique_ptr<bunch_type> pbase;
};



TEST_F(ParticleBaseTest, Create) {
    size_t nParticles = 1000;

    pbase->create(nParticles);

    size_t localnum = pbase->getLocalNum();

    EXPECT_EQ(nParticles, localnum);
}


TEST_F(ParticleBaseTest, Destroy) {
    size_t nParticles = 1000;
    size_t nDestroy   = 500;

    pbase->create(nParticles);

    using HostMirror = typename bunch_type::particle_index_type::HostMirror;
    HostMirror ID_host = Kokkos::create_mirror(pbase->ID);

    Kokkos::deep_copy(ID_host, pbase->ID);

    for (size_t i = 0; i < nDestroy; ++i) {
        ID_host(i) = -1;
    }

    Kokkos::deep_copy(pbase->ID, ID_host);

    pbase->destroy();

    size_t localnum = pbase->getLocalNum();

    EXPECT_EQ(nParticles - nDestroy, localnum);
}


TEST_F(ParticleBaseTest, AddAttribute) {

    using attrib_type = ippl::ParticleAttrib<double>;

    attrib_type Q;

    pbase->addAttribute(Q);

    auto nAttributes = pbase->getAttributeNum();

    EXPECT_EQ(3, nAttributes);
}







int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}