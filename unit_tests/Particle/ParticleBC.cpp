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

template <typename T>
class ParticleBCTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using playout_type = ippl::detail::ParticleLayout<T, Dim>;

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
    void setup(const ippl::Vector<T, Dim>& pos) {
        auto bunch = std::get<Idx>(bunches) =
            std::make_shared<bunch_type<Dim>>(std::get<Idx>(playouts));

        bunch->create(nParticles);

        auto& HostR = std::get<Idx>(mirrors) = bunch->R.getHostMirror();

        for (int i = 0; i < nParticles; ++i) {
            HostR(i) = pos;
        }

        Kokkos::deep_copy(bunch->R.getView(), HostR);

        // domain
        std::array<ippl::PRegion<T>, Dim> args;
        for (unsigned d = 0; d < Dim; d++) {
            args[d] = ippl::PRegion<T>(0, len[d]);
        }
        std::get<Idx>(nrs) = std::make_from_tuple<ippl::NDRegion<T, Dim>>(args);
    }

    template <unsigned Idx, unsigned Dim>
    void checkResult(const ippl::Vector<T, Dim>& expected) {
        T tol       = (std::is_same_v<T, double>) ? 1e-15 : 1e-7;
        auto& HostR = std::get<Idx>(mirrors);
        auto bunch  = std::get<Idx>(bunches);

        Kokkos::deep_copy(HostR, bunch->R.getView());

        for (int i = 0; i < nParticles; ++i) {
            for (size_t j = 0; j < Dim; ++j) {
                ASSERT_NEAR(expected[j], HostR(i)[j], tol);
            }
        }
    }

    template <unsigned Dim>
    ippl::Vector<T, Dim> truncate(const ippl::Vector<T, MaxDim>& vec) {
        ippl::Vector<T, Dim> res;
        for (unsigned d = 0; d < Dim; d++) {
            res[d] = vec[d];
        }
        return res;
    }

    PtrCollection<std::shared_ptr, bunch_type> bunches;

    ippl::Vector<T, MaxDim> len;
    ippl::Vector<T, MaxDim> shift;
    int nParticles;

    template <unsigned Dim>
    using region_type = ippl::NDRegion<T, Dim>;
    Collection<region_type> nrs;

    template <unsigned Dim>
    using mirror_type = typename bunch_type<Dim>::particle_position_type::HostMirror;
    Collection<mirror_type> mirrors;

    Collection<playout_type> playouts;
};

using Precisions = ::testing::Types<double, float>;

TYPED_TEST_CASE(ParticleBCTest, Precisions);

TYPED_TEST(ParticleBCTest, UpperPeriodicBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos =
            this->template truncate<Dim>(this->len + this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::PERIODIC);

        bunch->getLayout().applyBC(bunch->R, nr);

        auto expected = this->template truncate<Dim>(this->shift);
        this->template checkResult<Idx, Dim>(expected);
    };

    this->apply(check, this->bunches, this->nrs);
}

TYPED_TEST(ParticleBCTest, UpperNoBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos =
            this->template truncate<Dim>(this->len + this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::NO);

        bunch->getLayout().applyBC(bunch->R, nr);

        this->template checkResult<Idx, Dim>(pos);
    };

    this->apply(check, this->bunches, this->nrs);
}

TYPED_TEST(ParticleBCTest, UpperReflectiveBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos =
            this->template truncate<Dim>(this->len + this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::REFLECTIVE);

        bunch->getLayout().applyBC(bunch->R, nr);

        auto expected = this->template truncate<Dim>(this->len - this->shift);
        this->template checkResult<Idx, Dim>(expected);
    };

    this->apply(check, this->bunches, this->nrs);
}

TYPED_TEST(ParticleBCTest, UpperSinkBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos =
            this->template truncate<Dim>(this->len + this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::SINK);

        bunch->getLayout().applyBC(bunch->R, nr);

        this->template checkResult<Idx, Dim>(this->len);
    };

    this->apply(check, this->bunches, this->nrs);
}

TYPED_TEST(ParticleBCTest, LowerPeriodicBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx                  = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos = this->template truncate<Dim>(-this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::PERIODIC);

        bunch->getLayout().applyBC(bunch->R, nr);

        auto expected = this->template truncate<Dim>(this->len - this->shift);
        this->template checkResult<Idx, Dim>(expected);
    };

    this->apply(check, this->bunches, this->nrs);
}

TYPED_TEST(ParticleBCTest, LowerNoBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx                  = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos = this->template truncate<Dim>(-this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::NO);

        bunch->getLayout().applyBC(bunch->R, nr);

        this->template checkResult<Idx, Dim>(pos);
    };

    this->apply(check, this->bunches, this->nrs);
}

TYPED_TEST(ParticleBCTest, LowerReflectiveBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx                  = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos = this->template truncate<Dim>(-this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::REFLECTIVE);

        bunch->getLayout().applyBC(bunch->R, nr);

        ippl::Vector<TypeParam, Dim> expected = this->shift;
        this->template checkResult<Idx, Dim>(expected);
    };

    this->apply(check, this->bunches, this->nrs);
}

TYPED_TEST(ParticleBCTest, LowerSinkBC) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   ippl::NDRegion<TypeParam, Dim>& nr) {
        constexpr unsigned Idx                  = this->dimToIndex(Dim);
        const ippl::Vector<TypeParam, Dim>& pos = this->template truncate<Dim>(-this->shift);
        this->template setup<Idx, Dim>(pos);

        bunch->setParticleBC(ippl::BC::SINK);

        bunch->getLayout().applyBC(bunch->R, nr);

        ippl::Vector<TypeParam, Dim> expected = 0;
        this->template checkResult<Idx, Dim>(expected);
    };

    this->apply(check, this->bunches, this->nrs);
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
