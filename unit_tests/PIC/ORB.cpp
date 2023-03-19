//
// Unit tests ORB for class OrthogonalRecursiveBisection
//   Test volume and charge conservation in PIC operations.
//
// Copyright (c) 2021, Michael Ligotino, ETH, Zurich;
// Paul Scherrer Institut, Villigen; Switzerland
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

class ORBTest : public ::testing::Test {
public:
    static constexpr size_t dim = 3;
    using Mesh_t = ippl::UniformCartesian<double, dim>;
    using Centering_t = Mesh_t::DefaultCentering;
    using Field_t = ippl::Field<double, dim, Mesh_t, Centering_t>;
    using Flayout_t = ippl::FieldLayout<dim>;
    using Playout_t = ippl::ParticleSpatialLayout<double, dim>;
    using Orb_t = ippl::OrthogonalRecursiveBisection<double, dim, Mesh_t, Centering_t>;

    template <class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(Q);
        }

        ~Bunch() {}

        typedef ippl::ParticleAttrib<double> charge_container_type;
        charge_container_type Q;

        void updateLayout(Flayout_t fl, Mesh_t mesh) {
            PLayout& layout = this->getLayout();
            layout.updateLayout(fl, mesh);
        }
    };

    typedef Bunch<Playout_t> bunch_type;

    ORBTest()
        // Original configuration 256^3 particles, 512^3 grid.
        : nParticles(17000000)
        , nPoints(256) {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = ippl::PARALLEL;

        const bool isAllPeriodic = true;
        layout_m                 = Flayout_t(owned, allParallel, isAllPeriodic);

        double dx                        = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx     = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};

        mesh_m = Mesh_t(owned, hx, origin);

        field = std::make_unique<Field_t>(mesh_m, layout_m);

        pl_m = Playout_t(layout_m, mesh_m);

        bunch = std::make_unique<bunch_type>(pl_m);

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
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < nloc; ++i) {
            for (int d = 0; d < 3; d++) {
                R_host(i)[d] = unif(eng);
            }
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);

        orb.initialize(layout_m, mesh_m, *field);
    }

    void repartition() {
        bool fromAnalyticDensity = false;
        orb.binaryRepartition(bunch->R, layout_m, fromAnalyticDensity);
        field->updateLayout(layout_m);
        bunch->updateLayout(layout_m, mesh_m);
    }

    ippl::NDIndex<dim> getDomain() { return layout_m.getDomain(); }

    std::unique_ptr<Field_t> field;
    std::unique_ptr<bunch_type> bunch;
    size_t nParticles;
    size_t nPoints;

    Flayout_t layout_m;
    Mesh_t mesh_m;
    Playout_t pl_m;
    Orb_t orb;
};

TEST_F(ORBTest, Volume) {
    ippl::NDIndex<dim> dom = getDomain();
    bunch_type buffer(pl_m);

    pl_m.update(*bunch, buffer);

    repartition();

    pl_m.update(*bunch, buffer);

    ippl::NDIndex<dim> ndom = getDomain();

    ASSERT_DOUBLE_EQ(dom[0].length() * dom[1].length() * dom[2].length(),
                     ndom[0].length() * ndom[1].length() * ndom[2].length());
}

TEST_F(ORBTest, Charge) {
    bunch_type buffer(pl_m);

    double charge = 0.5;

    bunch->Q = charge;

    pl_m.update(*bunch, buffer);

    repartition();

    pl_m.update(*bunch, buffer);

    *field = 0.0;
    scatter(bunch->Q, *field, bunch->R);

    double totalCharge = field->sum();

    ASSERT_NEAR((nParticles * charge - totalCharge) / totalCharge, 0., 1e-13);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
