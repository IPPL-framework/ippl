//
// Unit test PICTest
//   Test scatter and gather particle-in-cell operations.
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

#include <random>

class PICTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::Field<double, dim> field_type;
    typedef ippl::FieldLayout<dim> flayout_type;
    typedef ippl::UniformCartesian<double, dim> mesh_type;
    typedef ippl::detail::ParticleLayout<double, dim> playout_type;

    template<class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout>
    {
        Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout)
        {
            this->addAttribute(Q);
        }
        typedef ippl::ParticleAttrib<double> charge_container_type;
        charge_container_type Q;
    };


    typedef Bunch<playout_type> bunch_type;


    PICTest()
    : nParticles(1000000)
    , nPoints(100)
    {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = ippl::SERIAL;

        layout_m = flayout_type (owned, allParallel);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};

        mesh_m = mesh_type(owned, hx, origin);

        field = std::make_unique<field_type>(mesh_m, layout_m);

        bunch = std::make_unique<bunch_type>(pl_m);

        bunch->create(nParticles);
        
        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = bunch->R.getHostMirror();
        for(size_t i = 0; i < nParticles; ++i) {
            ippl::Vector<double, dim> r = {unif(eng), unif(eng), unif(eng)};
            R_host(i) = r;
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }



    std::unique_ptr<field_type> field;
    std::unique_ptr<bunch_type> bunch;
    size_t nParticles;
    size_t nPoints;

private:
    flayout_type layout_m;
    mesh_type mesh_m;
    playout_type pl_m;
};



TEST_F(PICTest, Scatter) {

    *field = 0.0;

    double charge = 0.5;

    bunch->Q = charge;

    scatter(bunch->Q, *field, bunch->R);

    double totalcharge = field->sum(1);

    ASSERT_DOUBLE_EQ(nParticles * charge, totalcharge);

}

TEST_F(PICTest, Gather) {

    *field = 1.0;

    bunch->Q = 0.0;

    gather(bunch->Q, *field, bunch->R);

    ASSERT_DOUBLE_EQ(nParticles, bunch->Q.sum());

}

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
