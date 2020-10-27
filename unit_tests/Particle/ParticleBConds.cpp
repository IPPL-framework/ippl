//
// Unit test ParticleBCondsTest
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

#include <cmath>
#include "gtest/gtest.h"

class ParticleBCondsTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::detail::ParticleLayout<double, dim> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    ParticleBCondsTest() : len(0.2), shift(0.1) {
        setup();
    }

    void setup() {
        bunch = std::make_unique<bunch_type>(pl_m);

        bunch->create(1);

        HostR = Kokkos::create_mirror(bunch->R);

        HostR(0) = ippl::Vector<double, dim>({
            len + shift,
            len + shift,
            len + shift
        });

        Kokkos::deep_copy(bunch->R, HostR);

        // domain
        PRegion<double> region(0, len);
        nr = NDRegion<double, dim>(region, region, region);

    }

    std::unique_ptr<bunch_type> bunch;
    double len;
    double shift;
    NDRegion<double, dim>  nr;
    typename bunch_type::particle_position_type::HostMirror HostR;

private:
    playout_type pl_m;
};



TEST_F(ParticleBCondsTest, PeriodicBC) {

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticlePeriodicBCond<double>, i);
    }

    bunch->getLayout().applyBC(bunch->R, nr);

    Kokkos::deep_copy(HostR, bunch->R);

    ippl::Vector<double, dim> expected = {shift, shift, shift};

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], HostR(0)[i]);
    }
}


TEST_F(ParticleBCondsTest, NoBC) {

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticleNoBCond<double>, i);
    }

    bunch->getLayout().applyBC(bunch->R, nr);

    Kokkos::deep_copy(HostR, bunch->R);

    ippl::Vector<double, dim> expected = {
        len + shift,
        len + shift,
        len + shift
    };

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], HostR(0)[i]);
    }
}


TEST_F(ParticleBCondsTest, ReflectiveBC) {

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticleReflectiveBCond<double>, i);
    }

    bunch->getLayout().applyBC(bunch->R, nr);

    Kokkos::deep_copy(HostR, bunch->R);

    ippl::Vector<double, dim> expected = {
        len - shift,
        len - shift,
        len - shift
    };

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], HostR(0)[i]);
    }
}


TEST_F(ParticleBCondsTest, SinkBC) {

    for (unsigned i = 0; i < 2 * dim; i++) {
        bunch->setBCond(ippl::ParticleSinkBCond<double>, i);
    }

    bunch->getLayout().applyBC(bunch->R, nr);

    Kokkos::deep_copy(HostR, bunch->R);

    ippl::Vector<double, dim> expected = {len, len, len};

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_DOUBLE_EQ(expected[i], HostR(0)[i]);
    }
}


int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}