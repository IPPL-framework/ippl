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

TEST_F(ParticleBaseTest, CreateAndDestroy) {
    if (Ippl::Comm->size() > 1) {
        std::cerr << "ParticleBaseTest::CreateAndDestroy test only works for one MPI rank!"
                  << std::endl;
        return;
    }
    size_t nParticles = 1000;

    // Create 1000 particles
    pbase->create(nParticles);

    size_t localnum = pbase->getLocalNum();

    EXPECT_EQ(nParticles, localnum);

    // Check that the right IDs are present
    auto mirror = pbase->ID.getHostMirror();
    Kokkos::deep_copy(mirror, pbase->ID.getView());
    for (size_t i = 0; i < mirror.extent(0); ++i) {
        EXPECT_EQ(mirror[i], i);
    }

    // Delete all the particles with odd indices
    // (i.e. mark as invalid)
    typedef typename ippl::detail::ViewType<bool, 1>::view_type bool_type;
    bool_type invalid("invalid", nParticles);
    auto mirror2 = Kokkos::create_mirror(invalid);
    for (size_t i = 0; i < 500; ++i) {
        mirror2(2 * i)     = false;
        mirror2(2 * i + 1) = true;
    }
    Kokkos::deep_copy(invalid, mirror2);
    pbase->destroy(invalid, 500);

    // Verify remaining indices
    Kokkos::deep_copy(mirror, pbase->ID.getView());
    for (size_t i = 0; i < 500; ++i) {
        // The even indices contain the original particles
        // The particles with odd indices are deleted and replaced
        // with particles with even indices (in ascending order w.r.t. index)
        int index = i % 2 == 0 ? i : 500 + (i - 1);
        EXPECT_EQ(mirror[i], index);
    }
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

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
