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

#include <cmath>

#include "gtest/gtest.h"

class ParticleBCTest : public ::testing::Test {
public:
    enum {
        dim = 3
    };
    typedef ippl::detail::ParticleLayout<double, dim> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    ParticleBCTest()
        : len(0.2)
        , nParticles(1000) {}

    void setup(double pos) {
        bunch = std::make_unique<bunch_type>(pl_m);

        bunch->create(nParticles);

        HostR = bunch->R.getHostMirror();

        for (int i = 0; i < nParticles; ++i)
            HostR(i) = ippl::Vector<double, dim>({pos, pos, pos});

        Kokkos::deep_copy(bunch->R.getView(), HostR);

        // domain
        ippl::PRegion<double> region(0, len);
        nr = ippl::NDRegion<double, dim>(region, region, region);
    }

    void checkResult(const ippl::Vector<double, dim>& expected) {
        Kokkos::deep_copy(HostR, bunch->R.getView());

        for (int i = 0; i < nParticles; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                EXPECT_DOUBLE_EQ(expected[j], HostR(i)[j]);
            }
        }
    }

    std::unique_ptr<bunch_type> bunch;
    double len;
    int nParticles;
    ippl::NDRegion<double, dim> nr;
    typename bunch_type::particle_position_type::HostMirror HostR;
    playout_type pl_m;
};

TEST_F(ParticleBCTest, UpperPeriodicBC) {
    double shift = 0.05;
    setup(len + shift);

    bunch->setParticleBC(ippl::BC::PERIODIC);

    bunch->getLayout().applyBC(bunch->R, nr);

    checkResult({shift, shift, shift});
}

TEST_F(ParticleBCTest, UpperNoBC) {
    double shift = 0.05;
    setup(len + shift);

    bunch->setParticleBC(ippl::BC::NO);

    bunch->getLayout().applyBC(bunch->R, nr);

    Kokkos::deep_copy(HostR, bunch->R.getView());

    checkResult({len + shift, len + shift, len + shift});
}

TEST_F(ParticleBCTest, UpperReflectiveBC) {
    double shift = 0.05;
    setup(len + shift);

    bunch->setParticleBC(ippl::BC::REFLECTIVE);

    bunch->getLayout().applyBC(bunch->R, nr);

    checkResult({len - shift, len - shift, len - shift});
}

TEST_F(ParticleBCTest, UpperSinkBC) {
    double shift = 0.05;
    setup(len + shift);

    bunch->setParticleBC(ippl::BC::SINK);

    bunch->getLayout().applyBC(bunch->R, nr);

    checkResult({len, len, len});
}

TEST_F(ParticleBCTest, LowerPeriodicBC) {
    double shift = 0.05;
    setup(-shift);

    bunch->setParticleBC(ippl::BC::PERIODIC);

    bunch->getLayout().applyBC(bunch->R, nr);

    checkResult({len - shift, len - shift, len - shift});
}

TEST_F(ParticleBCTest, LowerNoBC) {
    double shift = 0.05;
    setup(-shift);

    bunch->setParticleBC(ippl::BC::NO);

    bunch->getLayout().applyBC(bunch->R, nr);

    checkResult({-shift, -shift, -shift});
}

TEST_F(ParticleBCTest, LowerReflectiveBC) {
    double shift = 0.05;
    setup(-shift);

    bunch->setParticleBC(ippl::BC::REFLECTIVE);

    bunch->getLayout().applyBC(bunch->R, nr);

    checkResult({shift, shift, shift});
}

TEST_F(ParticleBCTest, LowerSinkBC) {
    double shift = 0.05;
    setup(-shift);

    bunch->setParticleBC(ippl::BC::SINK);

    bunch->getLayout().applyBC(bunch->R, nr);

    checkResult({0, 0, 0});
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
