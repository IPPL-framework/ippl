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
    typedef FieldLayout<dim> flayout_type;
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
        Index I(nPoints);
        NDIndex<dim> owned(I, I, I);

        e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = SERIAL;

        layout_m = flayout_type (owned,allParallel, 1);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};

        mesh_m = mesh_type(owned, hx, origin);

        field = std::make_unique<field_type>(mesh_m, layout_m);

        bunch = std::make_unique<bunch_type>(pl_m);

        bunch->create(nParticles);
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
    std::mt19937_64 eng;
    std::uniform_real_distribution<double> unif(0, 1);

    double charge = 0.5;

    typename bunch_type::particle_position_type::HostMirror R_host = Kokkos::create_mirror(bunch->R);
    typename bunch_type::charge_container_type::HostMirror Q_host = Kokkos::create_mirror(bunch->Q);
    for(size_t i = 0; i < nParticles; ++i) {
        ippl::Vector<double, dim> r = {unif(eng), unif(eng), unif(eng)};
        R_host(i) = r;
        Q_host(i) = charge;
    }


    Kokkos::deep_copy(bunch->R, R_host);
    Kokkos::deep_copy(bunch->Q, Q_host);

    *field = 0.0;

    scatter(bunch->Q, *field, bunch->R);

    double totalcharge = field->sum(1);

    ASSERT_DOUBLE_EQ(nParticles * charge, totalcharge);
}




int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}