//
// Unit test ParticleBaseTest
//   Test functionality of the class ParticleBase.
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

#include <cmath>
#include "gtest/gtest.h"

class ParticleBaseTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::detail::ParticleLayout<double, dim> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    ParticleBaseTest() {
        setup();
    }

    void setup() {
        pbase = std::make_unique<bunch_type>(pl_m);
    }

    playout_type pl_m;
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

    EXPECT_EQ(size_t(3), nAttributes);
}


TEST(ParticleBase, Initialize1) {
    typedef ippl::detail::ParticleLayout<double, 3> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    playout_type pl;
    bunch_type bunch(pl);

    size_t localnum = bunch.getLocalNum();

    EXPECT_EQ(size_t(0), localnum);
}


TEST(ParticleBase, Initialize2) {
    typedef ippl::detail::ParticleLayout<double, 3> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    bunch_type bunch;

    playout_type pl;
    bunch.initialize(pl);

    size_t localnum = bunch.getLocalNum();

    EXPECT_EQ(size_t(0), localnum);
}




int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}