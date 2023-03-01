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
    typedef ippl::ParticleSpatialLayout<double, dim> playout_type;

    template <class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        Bunch(PLayout& playout) : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(Q);
        }

        ~Bunch() {
        }

        typedef ippl::ParticleAttrib<double> charge_container_type;
        charge_container_type Q;
    };

    typedef Bunch<playout_type> bunch_type;

    PICTest() : nParticles(std::pow(256, 3)), nPoints(512) {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag domDec[dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            domDec[d] = ippl::PARALLEL;

        layout_m = flayout_type(owned, domDec);

        double dx                        = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx     = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};

        mesh_m = mesh_type(owned, hx, origin);

        field = std::make_unique<field_type>(mesh_m, layout_m);

        pl = playout_type(layout_m, mesh_m);

        bunch = std::make_unique<bunch_type>(pl);

        int nRanks = Ippl::Comm->size();
        if (nParticles % nRanks > 0) {
            if (Ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
            exit(1);
        }

        size_t nloc = nParticles / nRanks;
        bunch->create(nloc);

        std::mt19937_64 eng;
        eng.seed(42);
        eng.discard(nloc * Ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(hx[0] / 2, 1 - (hx[0] / 2));

        typename bunch_type::particle_position_type::HostMirror R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < nloc; ++i) {
            for (int d = 0; d < 3; d++) {
                R_host(i)[d] = unif(eng);
            }
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    std::unique_ptr<field_type> field;
    std::unique_ptr<bunch_type> bunch;
    size_t nParticles;
    size_t nPoints;
    playout_type pl;

private:
    flayout_type layout_m;
    mesh_type mesh_m;
};

TEST_F(PICTest, Scatter) {
    *field = 0.0;

    double charge = 0.5;

    bunch->Q = charge;

    bunch_type bunchBuffer(pl);
    pl.update(*bunch, bunchBuffer);

    scatter(bunch->Q, *field, bunch->R);

    double totalcharge = field->sum();

    ASSERT_NEAR((nParticles * charge - totalcharge) / (nParticles * charge), 0.0, 1e-13);
}

TEST_F(PICTest, Gather) {
    *field = 1.0;

    bunch->Q = 0.0;

    bunch_type bunchBuffer(pl);
    pl.update(*bunch, bunchBuffer);

    gather(bunch->Q, *field, bunch->R);

    ASSERT_DOUBLE_EQ((nParticles - bunch->Q.sum()) / nParticles, 0.0);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
