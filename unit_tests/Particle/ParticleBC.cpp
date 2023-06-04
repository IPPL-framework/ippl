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

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class ParticleBCTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using playout_type = ippl::detail::ParticleLayout<double, Dim>;

    template <unsigned Dim>
    using bunch_type = ippl::ParticleBase<playout_type<Dim>>;

    ParticleBCTest()
        : nParticles(1000) {
        for (unsigned d = 0; d < MaxDim; d++) {
            len[d] = (d + 1) * 0.2;
        }
        for (unsigned d = 0; d < MaxDim; d++) {
            shift[d] = 0.01 * (d + 1);
        }
    }

    template <unsigned Idx, unsigned Dim>
    void setup(const ippl::Vector<double, Dim>& pos) {
        auto bunch = std::get<Idx>(bunches) =
            std::make_shared<bunch_type<Dim>>(std::get<Idx>(playouts));

        bunch->create(nParticles);

        auto& HostR = std::get<Idx>(mirrors) = bunch->R.getHostMirror();

        for (int i = 0; i < nParticles; ++i) {
            HostR(i) = pos;
        }

        Kokkos::deep_copy(bunch->R.getView(), HostR);

        // domain
        std::array<ippl::PRegion<double>, Dim> args;
        for (unsigned d = 0; d < Dim; d++) {
            args[d] = ippl::PRegion<double>(0, len[d]);
        }
        std::get<Idx>(nrs) = std::make_from_tuple<ippl::NDRegion<double, Dim>>(args);
    }

    template <unsigned Idx, unsigned Dim>
    void checkResult(const ippl::Vector<double, Dim>& expected) {
        auto& HostR = std::get<Idx>(mirrors);
        auto bunch  = std::get<Idx>(bunches);

        Kokkos::deep_copy(HostR, bunch->R.getView());

        for (int i = 0; i < nParticles; ++i) {
            for (size_t j = 0; j < Dim; ++j) {
                ASSERT_NEAR(expected[j], HostR(i)[j], 1e-15);
            }
        }
    }

    template <unsigned Dim>
    ippl::Vector<double, Dim> truncate(const ippl::Vector<double, MaxDim>& vec) {
        ippl::Vector<double, Dim> res;
        for (unsigned d = 0; d < Dim; d++) {
            res[d] = vec[d];
        }
        return res;
    }

    PtrCollection<std::shared_ptr, bunch_type> bunches;

    ippl::Vector<double, MaxDim> len;
    ippl::Vector<double, MaxDim> shift;
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
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(len + shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::PERIODIC);

        bunch->getLayout().applyBC(bunch->R, nr);

        auto expected = truncate<Dim>(shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, UpperNoBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(len + shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::NO);

        bunch->getLayout().applyBC(bunch->R, nr);

        checkResult<Idx, Dim>(pos);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, UpperReflectiveBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(len + shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::REFLECTIVE);

        bunch->getLayout().applyBC(bunch->R, nr);

        auto expected = truncate<Dim>(len - shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, UpperSinkBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(len + shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::SINK);

        bunch->getLayout().applyBC(bunch->R, nr);

        checkResult<Idx, Dim>(len);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerPeriodicBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(-shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::PERIODIC);

        bunch->getLayout().applyBC(bunch->R, nr);

        auto expected = truncate<Dim>(len - shift);
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerNoBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(-shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::NO);

        bunch->getLayout().applyBC(bunch->R, nr);

        checkResult<Idx, Dim>(pos);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerReflectiveBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(-shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::REFLECTIVE);

        bunch->getLayout().applyBC(bunch->R, nr);

        ippl::Vector<double, Dim> expected = shift;
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

TEST_F(ParticleBCTest, LowerSinkBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<double, Dim>& nr) {
        constexpr unsigned Idx = dimToIndex(Dim);
        auto pos               = truncate<Dim>(-shift);
        setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::SINK);

        bunch->getLayout().applyBC(bunch->R, nr);

        ippl::Vector<double, Dim> expected = 0;
        checkResult<Idx, Dim>(expected);
    };

    apply(check, bunches, nrs);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    ippl::finalize();
    return RUN_ALL_TESTS();
}
