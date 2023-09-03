//
// Unit test ParticleBaseTest
//   Test functionality of the class ParticleBase.
//
#include "Ippl.h"

#include "Particle/ParticleAttrib.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class ParticleBaseTest;
template <typename Params>
class InitializationTest : public ::testing::Test {
public:
    using Layout_t   = typename ParticleBaseTest<Params>::Layout_t;
    using bunch_type = typename ParticleBaseTest<Params>::bunch_type;
};

template <typename T, typename PositionSpace, unsigned Dim>
class ParticleBaseTest<Parameters<T, PositionSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type     = T;
    using attribute_type = ippl::ParticleAttrib<T, PositionSpace>;
    using bool_type      = typename ippl::detail::ViewType<bool, 1, PositionSpace>::view_type;
    using Layout_t       = typename ippl::ParticleBase<T, Dim, true, PositionSpace>::Layout_t;

    using bunch_type = ippl::ParticleBase<T, Dim, true, PositionSpace>;

    ParticleBaseTest()
        : pbase(std::make_shared<bunch_type>(rlayout)) {}

    Layout_t rlayout;
    std::shared_ptr<bunch_type> pbase;
};

using Precisions = TestParams::Precisions;
using Spaces     = TestParams::Spaces;
using Ranks      = TestParams::Ranks<1, 2, 3, 4, 5, 6>;
using Combos     = CreateCombinations<Precisions, Spaces, Spaces, Ranks>::type;
using Tests      = TestForTypes<Combos>::type;
TYPED_TEST_CASE(ParticleBaseTest, Tests);
TYPED_TEST_CASE(InitializationTest, Tests);

TYPED_TEST(ParticleBaseTest, CreateAndDestroy) {
    if (ippl::Comm->size() > 1) {
        std::cerr << "ParticleBaseTest::CreateAndDestroy test only works for one MPI rank!"
                  << std::endl;
        return;
    }
    size_t nParticles = 1000;

    auto& pbase = this->pbase;

    // Create 1000 particles
    pbase->create(nParticles);

    size_t localnum = pbase->getLocalNum();

    EXPECT_EQ(nParticles, localnum);

    // Check that the right IDs are present
    auto mirror = pbase->ID.getHostMirror();
    Kokkos::deep_copy(mirror, pbase->ID.getView());
    for (size_t i = 0; i < mirror.extent(0); ++i) {
        EXPECT_EQ(mirror[i], (int)i);
    }

    // Delete all the particles with odd indices
    // (i.e. mark as invalid)
    typename TestFixture::bool_type invalid("invalid", nParticles);
    auto mirror2 = Kokkos::create_mirror(invalid);
    for (size_t i = 0; i < 500; ++i) {
        mirror2(2 * i)     = false;
        mirror2(2 * i + 1) = true;
    }
    Kokkos::deep_copy(invalid, mirror2);
    pbase->destroy(invalid, 500);

    // Verify remaining indices
    Kokkos::deep_copy(mirror, pbase->ID.getView());
    for (int i = 0; i < 500; ++i) {
        // The even indices contain the original particles
        // The particles with odd indices are deleted and replaced
        // with particles with even indices (in ascending order w.r.t. index)
        int index = i % 2 == 0 ? i : 500 + (i - 1);
        EXPECT_EQ(mirror[i], index);
    }
}

TYPED_TEST(ParticleBaseTest, AddAttribute) {
    using attrib_type = typename TestFixture::attribute_type;

    auto& pbase = this->pbase;

    attrib_type Q;

    pbase->addAttribute(Q);

    auto nAttributes = pbase->getAttributeNum();

    EXPECT_EQ(size_t(3), nAttributes);
}

TYPED_TEST(InitializationTest, Initialize1) {
    typename TestFixture::Layout_t pl;
    typename TestFixture::bunch_type bunch(pl);

    size_t localnum = bunch.getLocalNum();

    EXPECT_EQ(size_t(0), localnum);
}

TYPED_TEST(InitializationTest, Initialize2) {
    typename TestFixture::Layout_t pl;
    typename TestFixture::bunch_type bunch;

    bunch.initialize(pl);

    size_t localnum = bunch.getLocalNum();

    EXPECT_EQ(size_t(0), localnum);
}

int main(int argc, char* argv[]) {
    int success = 1;
    TestParams::checkArgs(argc, argv);
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
