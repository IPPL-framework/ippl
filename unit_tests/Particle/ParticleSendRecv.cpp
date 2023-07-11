//
// Unit test Particle send/receive
//   Test particle send and receive operations.
//
// Copyright (c) 2020, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include <random>

#include "MultirankUtils.h"
#include "gtest/gtest.h"

template <typename T>
class ParticleSendRecv : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using flayout_type = ippl::FieldLayout<Dim>;

    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<T, Dim>;

    template <unsigned Dim>
    using playout_type = ippl::ParticleSpatialLayout<T, Dim>;

    template <unsigned Dim>
    using RegionLayout_t = ippl::detail::RegionLayout<T, Dim, mesh_type<Dim>>::uniform_type;

    typedef ippl::ParticleAttrib<int> rank_type;

    template <class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(expectedRank);
            this->addAttribute(Q);
        }

        ~Bunch() {}

        typedef ippl::ParticleAttrib<int> rank_type;
        typedef ippl::ParticleAttrib<T> charge_container_type;
        
        rank_type expectedRank;
        charge_container_type Q;

        void update() {
            PLayout& layout = this->getLayout();
            layout.update(*this);
        }
    };

    template <unsigned Dim>
    using bunch_type = Bunch<playout_type<Dim>>;

    ParticleSendRecv()
        : nParticles(128) {
        computeGridSizes(nPoints);
        for (unsigned d = 0; d < MaxDim; d++) {
            domain[d] = nPoints[d] / 16.;
        }
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++) {
            args[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < Dim; d++) {
            domDec[d] = ippl::PARALLEL;
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        auto& layout = std::get<Idx>(layouts) = flayout_type<Dim>(owned, domDec);
        auto& mesh = std::get<Idx>(meshes) = mesh_type<Dim>(owned, hx, origin);
        auto& pl = std::get<Idx>(playouts) = playout_type<Dim>(layout, mesh);
        auto bunch = std::get<Idx>(bunches) = std::make_shared<bunch_type<Dim>>(pl);

        using BC = ippl::BC;

        typename bunch_type<Dim>::bc_container_type bcs;
        bcs.fill(BC::PERIODIC);

        bunch->setParticleBC(bcs);

        int nRanks = ippl::Comm->size();
        if (nParticles % nRanks > 0) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
        }

        bunch->create(nParticles / nRanks);

        std::mt19937_64 eng(ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(0, 1);

        auto R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            ippl::Vector<T, Dim> r;
            for (unsigned d = 0; d < Dim; d++) {
                r[d] = unif(eng) * domain[d];
            }
            R_host(i) = r;
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
        bunch->Q                    = 1.0;
        RegionLayout_t<Dim> RLayout = pl.getRegionLayout();

        using region_view  = typename RegionLayout_t<Dim>::view_type;
        using size_type    = typename RegionLayout_t<Dim>::view_type::size_type;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        auto& positions         = bunch->R.getView();
        region_view Regions     = RLayout.getdLocalRegions();
        rank_type::view_type ER = bunch->expectedRank.getView();

        Kokkos::parallel_for(
            "Expected Rank", mdrange_type({0, 0}, {ER.extent(0), Regions.extent(0)}),
            KOKKOS_LAMBDA(const size_t i, const size_type j) {
                bool xyz_bool = true;
                for (unsigned d = 0; d < Dim; d++) {
                    xyz_bool &= positions(i)[d] <= Regions(j)[d].max()
                                && positions(i)[d] >= Regions(j)[d].min();
                }
                if (xyz_bool) {
                    ER(i) = j;
                }
            });
        Kokkos::fence();
    }

    PtrCollection<std::shared_ptr, bunch_type> bunches;
    unsigned int nParticles;
    size_t nPoints[MaxDim];
    T domain[MaxDim];
    Collection<playout_type> playouts;

private:
    Collection<flayout_type> layouts;
    Collection<mesh_type> meshes;
};

using Precisions = ::testing::Types<double, float>;

TYPED_TEST_CASE(ParticleSendRecv, Precisions);

TYPED_TEST(ParticleSendRecv, SendAndRecieve) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::bunch_type<Dim>>& bunch,
                                   typename TestFixture::playout_type<Dim>& pl) {
        typename TestFixture::bunch_type<Dim> bunchBuffer(pl);
        pl.update(*bunch, bunchBuffer);
        // bunch->update();
        typename TestFixture::ER_t::view_type::host_mirror_type ER_host =
            bunch->expectedRank.getHostMirror();
            
        Kokkos::resize(ER_host, bunch->expectedRank.size());
        Kokkos::deep_copy(ER_host, bunch->expectedRank.getView());

        for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
            ASSERT_EQ(ER_host(i), ippl::Comm->rank());
        }
        ippl::Comm->barrier();

        unsigned int Total_particles = 0;
        unsigned int local_particles = bunch->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, MPI_UNSIGNED, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            ASSERT_EQ(this->nParticles, Total_particles);
        }
    };

    this->apply(check, this->bunches, this->playouts);
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
