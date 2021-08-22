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

#include <cmath>
#include "gtest/gtest.h"

#include <random>

class ParticleSendRecv : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::FieldLayout<dim> flayout_type;
    typedef ippl::UniformCartesian<double, dim> mesh_type;
    typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
    typedef ippl::detail::RegionLayout<double, 3, mesh_type> RegionLayout_t;
    typedef ippl::ParticleAttrib<int> ER_t;

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
        typedef ippl::ParticleAttrib<double> charge_container_type;
        rank_type expectedRank;
        charge_container_type Q;
    
        void update() {
            PLayout& layout = this->getLayout();
            layout.update(*this);
        }
    };


    typedef Bunch<playout_type> bunch_type;


    ParticleSendRecv()
    : nParticles(std::pow(256,3))
    , nPoints(1024)
    {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag domDec[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            domDec[d] = ippl::PARALLEL;

        layout_m = flayout_type(owned, domDec);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};

        mesh_m = mesh_type(owned, hx, origin);

        pl = playout_type(layout_m, mesh_m);
        
        bunch = std::make_unique<bunch_type>(pl);
        
        using BC = ippl::BC;

        bunch_type::bc_container_type bcs = {
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC,
            BC::PERIODIC
        };

        bunch->setParticleBC(bcs);

        int nRanks = Ippl::Comm->size();
        if (nParticles % nRanks > 0) {
            if (Ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
        }

        bunch->create(nParticles / nRanks);
        
        std::mt19937_64 eng(Ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = bunch->R.getHostMirror();
        for(size_t i = 0; i < bunch->getLocalNum(); ++i) {
            ippl::Vector<double, dim> r = {unif(eng), unif(eng), unif(eng)};
            R_host(i) = r;
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
        bunch->Q = 1.0;
        RegionLayout_t RLayout = pl.getRegionLayout();

        auto& positions = bunch->R.getView();
        typename RegionLayout_t::view_type Regions = RLayout.getdLocalRegions();
        using size_type = typename RegionLayout_t::view_type::size_type;
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        ER_t::view_type ER = bunch->expectedRank.getView();

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
    }

    std::unique_ptr<bunch_type> bunch;
    unsigned int nParticles;
    size_t nPoints;
    playout_type pl;

private:
    flayout_type layout_m;
    mesh_type mesh_m;
};



TEST_F(ParticleSendRecv, SendAndRecieve) {

    bunch_type bunchBuffer(pl);
    pl.update(*bunch, bunchBuffer);
    //bunch->update();
    ER_t::view_type::host_mirror_type ER_host = bunch->expectedRank.getHostMirror();
    Kokkos::resize(ER_host, bunch->expectedRank.size());
    Kokkos::deep_copy(ER_host, bunch->expectedRank.getView());

    for (size_t i = 0; i < bunch->getLocalNum(); ++i) {
        ASSERT_EQ(ER_host(i), Ippl::Comm->rank());
    }
    Ippl::Comm->barrier();

    unsigned int Total_particles = 0;
    unsigned int local_particles = bunch->getLocalNum();

    MPI_Reduce(&local_particles, &Total_particles, 1, 
                MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());
    
    if (Ippl::Comm->rank() == 0) {

        ASSERT_EQ(nParticles, Total_particles);
    }



}


int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
