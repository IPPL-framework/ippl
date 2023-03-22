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
#include <random>

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class PICTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using field_type = ippl::Field<double, Dim>;

    template <unsigned Dim>
    using flayout_type = ippl::FieldLayout<Dim>;

    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<double, Dim>;

    template <unsigned Dim>
    using playout_type = ippl::ParticleSpatialLayout<double, Dim>;

    template <class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(Q);
        }

        ~Bunch() {}

        typedef ippl::ParticleAttrib<double> charge_container_type;
        charge_container_type Q;
    };

    template <unsigned Dim>
    using bunch_type = Bunch<playout_type<Dim>>;

    PICTest()
        : nParticles(32)
        , nPoints(16) {
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        ippl::Index I(nPoints);
        std::array<ippl::Index, Dim> args;
        args.fill(I);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < Dim; d++) {
            domDec[d] = ippl::PARALLEL;
            hx[d]     = dx;
            origin[d] = 0;
        }

        auto& layout = std::get<Idx>(layouts) = flayout_type<Dim>(owned, domDec);
        auto& mesh = std::get<Idx>(meshes) = mesh_type<Dim>(owned, hx, origin);

        std::get<Idx>(fields) = std::make_unique<field_type<Dim>>(mesh, layout);

        auto& pl = std::get<Idx>(playouts) = playout_type<Dim>(layout, mesh);

        auto& bunch = std::get<Idx>(bunches) = std::make_unique<bunch_type<Dim>>(pl);

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

        auto R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < nloc; ++i) {
            for (unsigned d = 0; d < Dim; d++) {
                R_host(i)[d] = unif(eng);
            }
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);
    }

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, bunch_type> bunches;
    size_t nParticles;
    size_t nPoints;
    Collection<playout_type> playouts;

private:
    Collection<flayout_type> layouts;
    Collection<mesh_type> meshes;
};

TEST_F(PICTest, Scatter) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field,
                                   std::shared_ptr<bunch_type<Dim>>& bunch, playout_type<Dim>& pl) {
        *field = 0.0;

        double charge = 0.5;

        bunch->Q = charge;

        bunch_type<Dim> bunchBuffer(pl);
        pl.update(*bunch, bunchBuffer);

        scatter(bunch->Q, *field, bunch->R);

        double totalcharge = field->sum();

        ASSERT_NEAR((nParticles * charge - totalcharge) / (nParticles * charge), 0.0, 1e-13);
    };

    auto args = zip(fields, bunches, playouts);
    apply(check, args);
}

TEST_F(PICTest, Gather) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field,
                                   std::shared_ptr<bunch_type<Dim>>& bunch, playout_type<Dim>& pl) {
        *field = 1.0;

        bunch->Q = 0.0;

        bunch_type<Dim> bunchBuffer(pl);
        pl.update(*bunch, bunchBuffer);

        gather(bunch->Q, *field, bunch->R);

        ASSERT_DOUBLE_EQ((nParticles - bunch->Q.sum()) / nParticles, 0.0);
    };

    auto args = zip(fields, bunches, playouts);
    apply(check, args);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
