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

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class ParticleBaseTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using playout_type = ippl::detail::ParticleLayout<double, Dim>;

    template <unsigned Dim>
    using bunch_type = ippl::ParticleBase<playout_type<Dim>, Kokkos::DefaultExecutionSpace>;

    ParticleBaseTest() { setup(this); }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        std::get<Idx>(pbases) = std::make_shared<bunch_type<Dim>>(std::get<Idx>(playouts));
    }

    Collection<playout_type> playouts;
    PtrCollection<std::shared_ptr, bunch_type> pbases;
};

TEST_F(ParticleBaseTest, CreateAndDestroy) {
    if (ippl::Comm->size() > 1) {
        std::cerr << "ParticleBaseTest::CreateAndDestroy test only works for one MPI rank!"
                  << std::endl;
        return;
    }
    size_t nParticles = 1000;

    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& pbase) {
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
    };

    apply(check, pbases);
}

TEST_F(ParticleBaseTest, AddAttribute) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& pbase) {
        using attrib_type = ippl::ParticleAttrib<double>;

        attrib_type Q;

        pbase->addAttribute(Q);

        auto nAttributes = pbase->getAttributeNum();

        EXPECT_EQ(size_t(3), nAttributes);
    };

    apply(check, pbases);
}

TEST(ParticleBase, Initialize1) {
    auto check = [&]<unsigned Dim>() {
        ParticleBaseTest::playout_type<Dim> pl;
        ParticleBaseTest::bunch_type<Dim> bunch(pl);

        size_t localnum = bunch.getLocalNum();

        EXPECT_EQ(size_t(0), localnum);
    };

    MultirankUtils<1, 2, 3, 4, 5, 6>::apply(check);
}

TEST(ParticleBase, Initialize2) {
    auto check = [&]<unsigned Dim>() {
        ParticleBaseTest::bunch_type<Dim> bunch;

        ParticleBaseTest::playout_type<Dim> pl;
        bunch.initialize(pl);

        size_t localnum = bunch.getLocalNum();

        EXPECT_EQ(size_t(0), localnum);
    };

    MultirankUtils<1, 2, 3, 4, 5, 6>::apply(check);
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
