//
// Unit test ParticleSpatialLayoutTest
//   Test functionality of the class ParticleSpatialLayout.
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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
#include <random>
#include "gtest/gtest.h"

template<class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout>
{

    Bunch(PLayout& playout)
    : ippl::ParticleBase<PLayout>(playout)
    {
        this->addAttribute(expectedRank);
        this->addAttribute(Q);
    }

    ~Bunch(){ }

    typedef ippl::ParticleAttrib<int> rank_type;
    typedef ippl::ParticleAttrib<double> charge_type;
    rank_type expectedRank;
    charge_type Q;
};

class ParticleSpatialLayoutTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    static constexpr int pt = 8;
    static constexpr int nParticles = 20480; // pt ** 3 * 40

    typedef ippl::ParticleSpatialLayout<double, dim> playout_type;
    typedef Bunch<playout_type> bunch_type;
    typedef ippl::FieldLayout<dim> layout_type;

    ParticleSpatialLayoutTest() {
        setup();
    }

    void setup() {
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d=0; d<dim; d++)
            allParallel[d] = ippl::PARALLEL;

        layout_m = std::make_shared<layout_type>(owned,allParallel);

        double dx = 1.0 / double(pt);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};
        typedef ippl::UniformCartesian<double, dim> Mesh_t;
        Mesh_t mesh(owned, hx, origin);

        pl_m = std::make_unique<playout_type>(*layout_m, mesh);

        pbase_m = std::make_unique<bunch_type>(*pl_m);

        using BC = ippl::BC;

        bunch_type::bc_container_type bcs = {
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC
        };

        pbase_m->setParticleBC(bcs);

        int nRanks = Ippl::Comm->size();

        if (nParticles % nRanks > 0) {
            if (Ippl::Comm->rank() == 0) {
                std::cerr << "Particle count " << nParticles <<
                    " not a multiple of rank count " << nRanks << std::endl;
            }
        }
        ASSERT_EQ(nParticles % nRanks, 0);

        pbase_m->create(nParticles / nRanks);

        #ifdef KOKKOS_ENABLE_CUDA
        int id = -1;
        auto err = cudaGetDevice(&id);
        if (err != cudaSuccess) printf("kernel cuda error: %d, %s\n", (int)err, cudaGetErrorString(err));
        std::cout << "Rank " << Ippl::Comm->rank() << " has device " << id << "\n";
        ASSERT_EQ(err, cudaSuccess);
        #endif

        std::mt19937_64 eng(Ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = pbase_m->R.getHostMirror();
        for (size_t i = 0; i < pbase_m->getLocalNum(); ++i) {
            ippl::Vector<double, dim> r = {unif(eng), unif(eng), unif(eng)};
            R_host(i) = r;
        }
        Ippl::Comm->barrier();
        Kokkos::deep_copy(pbase_m->R.getView(), R_host);
    }

    std::unique_ptr<playout_type> pl_m;
    std::unique_ptr<bunch_type> pbase_m;
    std::shared_ptr<layout_type> layout_m;
};

TEST_F(ParticleSpatialLayoutTest, Update) {
    typedef ippl::UniformCartesian<double, dim> Mesh_t;
    typedef ippl::detail::RegionLayout<double, dim, Mesh_t> RegionLayout_t;
    RegionLayout_t RLayout = pl_m->getRegionLayout();

    std::cout << *layout_m << std::endl;

    auto& positions = pbase_m->R.getView();
    typename RegionLayout_t::view_type Regions = RLayout.getdLocalRegions();
    using size_type = typename RegionLayout_t::view_type::size_type;
    using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    typedef ippl::ParticleAttrib<int> ER_t;
    ER_t::view_type ER = pbase_m->expectedRank.getView();

    Kokkos::parallel_for("Expected Rank",
            mdrange_type({0, 0},
                         {ER.extent(0), Regions.extent(0)}), 
            KOKKOS_LAMBDA(const size_t i, const size_type j) {
                bool x_bool = false;
                bool y_bool = false;
                bool z_bool = false;
                if((positions(i)[0] >= Regions(j)[0].min()) &&
                   (positions(i)[0] <= Regions(j)[0].max())) {
                    x_bool = true;    
                }
                if((positions(i)[1] >= Regions(j)[1].min()) &&
                   (positions(i)[1] <= Regions(j)[1].max())) {
                    y_bool = true;    
                }
                if((positions(i)[2] >= Regions(j)[2].min()) &&
                   (positions(i)[2] <= Regions(j)[2].max())) {
                    z_bool = true;    
                }
                if(x_bool && y_bool && z_bool){
                    ER(i) = j;
                }
        });
    Kokkos::fence();

    typename bunch_type::particle_index_type::HostMirror ID_host = pbase_m->ID.getHostMirror();
    Kokkos::deep_copy(ID_host, pbase_m->ID.getView());

    ER_t::view_type::host_mirror_type ER_host = pbase_m->expectedRank.getHostMirror();
    Kokkos::deep_copy(ER_host, pbase_m->expectedRank.getView());
    typedef ippl::ParticleAttrib<double> Q_t;
    Q_t::view_type::host_mirror_type Q_host = pbase_m->Q.getHostMirror();

    Kokkos::deep_copy(Q_host, pbase_m->Q.getView());

    bunch_type bunchBuffer(*pl_m);
    pl_m->update(*pbase_m, bunchBuffer);
    Ippl::Comm->barrier();

    typename bunch_type::particle_position_type::HostMirror R_host = pbase_m->R.getHostMirror();
    Kokkos::resize(R_host, pbase_m->R.size());
    Kokkos::deep_copy(R_host, pbase_m->R.getView());

    Kokkos::resize(ID_host, pbase_m->ID.size());
    Kokkos::deep_copy(ID_host, pbase_m->ID.getView());
    
    Kokkos::resize(Q_host, pbase_m->Q.size());
    Kokkos::deep_copy(Q_host, pbase_m->Q.getView());
    
    Kokkos::resize(ER_host, pbase_m->expectedRank.size());
    Kokkos::deep_copy(ER_host, pbase_m->expectedRank.getView());

    for (size_t i = 0; i < pbase_m->getLocalNum(); ++i) {
        EXPECT_EQ(Ippl::Comm->rank(), ER_host(i));
    }
    Ippl::Comm->barrier();

    unsigned int total_particles = 0;
    unsigned int local_particles = pbase_m->getLocalNum();

    MPI_Reduce(&local_particles, &total_particles, 1, 
                MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());
    if (Ippl::Comm->rank() == 0) {
        EXPECT_EQ(nParticles, total_particles);
    }
}

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
