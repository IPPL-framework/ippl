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

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class ParticleBCTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using playout_type = ippl::detail::ParticleLayout<double, Dim>;

    template <unsigned Dim>
    using bunch_type = ippl::ParticleBase<playout_type<Dim>>;

    ParticleBCTest()
        : len(0.2)
        , nParticles(1000) {}

    template <unsigned Idx, unsigned Dim>
    void setup(double pos) {
        auto bunch = std::get<Idx>(bunches) =
            std::make_shared<bunch_type<Dim>>(std::get<Idx>(playouts));

        bunch->create(nParticles);

        auto& HostR = std::get<Idx>(mirrors) = bunch->R.getHostMirror();

        for (int i = 0; i < nParticles; ++i)
            HostR(i) = ippl::Vector<double, Dim>(pos);

        Kokkos::deep_copy(bunch->R.getView(), HostR);

        // domain
        ippl::PRegion<double> region(0, len);
        std::array<ippl::PRegion<double>, Dim> args;
        args.fill(region);
        std::get<Idx>(nrs) = std::make_from_tuple<ippl::NDRegion<double, Dim>>(args);
    }

    template <unsigned Idx, unsigned Dim>
    void checkResult(const std::array<double, Dim>& expected) {
        auto& HostR = std::get<Idx>(mirrors);
        auto bunch  = std::get<Idx>(bunches);

        Kokkos::deep_copy(HostR, bunch->R.getView());

        for (int i = 0; i < nParticles; ++i) {
            for (size_t j = 0; j < Dim; ++j) {
                EXPECT_DOUBLE_EQ(expected[j], HostR(i)[j]);
            }
        }
    }

    PtrCollection<std::shared_ptr, bunch_type> bunches;

    double len;
    int nParticles;

    template <unsigned Dim>
    using region_type = ippl::NDRegion<double, Dim>;
    Collection<region_type> nrs;

    template <unsigned Dim>
    using mirror_type = typename bunch_type<Dim>::particle_position_type::HostMirror;
    Collection<mirror_type> mirrors;

    Collection<playout_type> playouts;
};

TEST_F(ParticleBCTest, UpperPeriodicBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(len + shift);

        bunch->setParticleBC(ippl::BC::PERIODIC);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, UpperNoBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(len + shift);

        bunch->setParticleBC(ippl::BC::NO);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(len + shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, UpperReflectiveBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(len + shift);

        bunch->setParticleBC(ippl::BC::REFLECTIVE);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(len - shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, UpperSinkBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(len + shift);

        bunch->setParticleBC(ippl::BC::SINK);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(len);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerPeriodicBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(-shift);

        bunch->setParticleBC(ippl::BC::PERIODIC);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(len - shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerNoBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(-shift);

        bunch->setParticleBC(ippl::BC::NO);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(-shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerReflectiveBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(-shift);

        bunch->setParticleBC(ippl::BC::REFLECTIVE);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerSinkBC) {
    double shift = 0.05;
    auto check   = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        setup<Idx, Dim>(-shift);

        bunch->setParticleBC(ippl::BC::SINK);

        bunch->getLayout().applyBC(bunch->R, nr);

        std::array<double, Dim> expected;
        expected.fill(0);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
