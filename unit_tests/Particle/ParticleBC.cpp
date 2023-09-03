//
// Unit test ParticleBCTest
//   Test particle boundary conditions.
//
#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class ParticleBCTest;

template <typename T, typename ExecSpace, unsigned Dim>
class ParticleBCTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
protected:
    void SetUp() override { CHECK_SKIP_SERIAL; }

public:
    using value_type              = T;
    constexpr static unsigned dim = Dim;

    using Layout_t   = typename ippl::ParticleBase<T, Dim, false, ExecSpace>::Layout_t;
    using bunch_type = ippl::ParticleBase<T, Dim, false, ExecSpace>;

    ParticleBCTest() {
        CHECK_SKIP_SERIAL_CONSTRUCTOR;
        for (unsigned d = 0; d < Dim; d++) {
            len[d] = (d + 1) * 0.2;
        }
        for (unsigned d = 0; d < Dim; d++) {
            shift[d] = 0.01 * (d + 1);
        }
    }

    void setup(const ippl::Vector<T, Dim>& pos) {
        bunch = std::make_shared<bunch_type>(rlayout);

        bunch->create(nParticles);

        mirror = bunch->R.getHostMirror();

        for (int i = 0; i < nParticles; ++i) {
            mirror(i) = pos;
        }

        Kokkos::deep_copy(bunch->R.getView(), mirror);

        // domain
        std::array<ippl::PRegion<T>, Dim> args;
        for (unsigned d = 0; d < Dim; d++) {
            args[d] = ippl::PRegion<T>(0, len[d]);
        }
        nr = std::make_from_tuple<ippl::NDRegion<T, Dim>>(args);
    }

    void checkResult(const ippl::Vector<T, Dim>& expected) {
        T tol = (std::is_same_v<T, double>) ? 1e-15 : 1e-7;

        Kokkos::deep_copy(mirror, bunch->R.getView());

        for (int i = 0; i < nParticles; ++i) {
            for (size_t j = 0; j < Dim; ++j) {
                ASSERT_NEAR(expected[j], mirror(i)[j], tol);
            }
        }
    }

    std::shared_ptr<bunch_type> bunch;

    ippl::Vector<T, Dim> len;
    ippl::Vector<T, Dim> shift;
    int nParticles = 1000;

    using region_type = ippl::NDRegion<T, Dim>;
    region_type nr;

    using mirror_type = typename bunch_type::particle_position_type::HostMirror;
    mirror_type mirror;

    Layout_t rlayout;
};

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_CASE(ParticleBCTest, Tests);

TYPED_TEST(ParticleBCTest, UpperPeriodicBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = this->len + this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::PERIODIC);

    bunch->applyBC(nr);

    auto expected = this->shift;
    this->checkResult(expected);
}

TYPED_TEST(ParticleBCTest, UpperNoBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = this->len + this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::NO);

    bunch->applyBC(nr);

    this->checkResult(pos);
}

TYPED_TEST(ParticleBCTest, UpperReflectiveBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = this->len + this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::REFLECTIVE);

    bunch->applyBC(nr);

    auto expected = this->len - this->shift;
    this->checkResult(expected);
}

TYPED_TEST(ParticleBCTest, UpperSinkBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = this->len + this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::SINK);

    bunch->applyBC(nr);

    this->checkResult(this->len);
}

TYPED_TEST(ParticleBCTest, LowerPeriodicBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = -this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::PERIODIC);

    bunch->applyBC(nr);

    auto expected = this->len - this->shift;
    this->checkResult(expected);
}

TYPED_TEST(ParticleBCTest, LowerNoBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = -this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::NO);

    bunch->applyBC(nr);

    this->checkResult(pos);
}

TYPED_TEST(ParticleBCTest, LowerReflectiveBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = -this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::REFLECTIVE);

    bunch->applyBC(nr);

    ippl::Vector<T, Dim> expected = this->shift;
    this->checkResult(expected);
}

TYPED_TEST(ParticleBCTest, LowerSinkBC) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& bunch = this->bunch;
    auto& nr    = this->nr;

    const ippl::Vector<T, Dim>& pos = -this->shift;
    this->setup(pos);

    bunch->setParticleBC(ippl::BC::SINK);

    bunch->applyBC(nr);

    ippl::Vector<T, Dim> expected = 0;
    this->checkResult(expected);
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
