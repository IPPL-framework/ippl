//
// Unit test ParticleBCTest
//   Test particle boundary conditions.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
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

    using playout_type = ippl::detail::ParticleLayout<T, Dim, ExecSpace>;
    using bunch_type   = ippl::ParticleBase<playout_type>;

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
        bunch = std::make_shared<bunch_type>(playout);

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

    playout_type playout;
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

    bunch->getLayout().applyBC(bunch->R, nr);

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

    bunch->getLayout().applyBC(bunch->R, nr);

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

    bunch->getLayout().applyBC(bunch->R, nr);

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

    bunch->getLayout().applyBC(bunch->R, nr);

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

    bunch->getLayout().applyBC(bunch->R, nr);

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

    bunch->getLayout().applyBC(bunch->R, nr);

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

    bunch->getLayout().applyBC(bunch->R, nr);

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

    bunch->getLayout().applyBC(bunch->R, nr);

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
