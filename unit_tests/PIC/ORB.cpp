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

#include <random>

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class ORBTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<double, Dim>;

    template <unsigned Dim>
    using centering_type = typename mesh_type<Dim>::DefaultCentering;

    template <unsigned Dim>
    using field_type = ippl::Field<double, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <unsigned Dim>
    using flayout_type = ippl::FieldLayout<Dim>;

    template <unsigned Dim>
    using playout_type = ippl::ParticleSpatialLayout<double, Dim>;

    template <unsigned Dim>
    using ORB =
        ippl::OrthogonalRecursiveBisection<double, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <class PLayout, unsigned Dim>
    struct Bunch : public ippl::ParticleBase<PLayout> {
        Bunch(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            this->addAttribute(Q);
        }

        ~Bunch() {}

        typedef ippl::ParticleAttrib<double> charge_container_type;
        charge_container_type Q;

        void updateLayout(flayout_type<Dim> fl, mesh_type<Dim> mesh) {
            PLayout& layout = this->getLayout();
            layout.updateLayout(fl, mesh);
        }
    };

    template <unsigned Dim>
    using bunch_type = Bunch<playout_type<Dim>, Dim>;

    ORBTest()
        // Original configuration 256^3 particles, 512^3 grid.
        : nParticles(128) {
        computeGridSizes(nPoints);
        for (unsigned d = 0; d < MaxDim; d++)
            domain[d] = nPoints[d] / 32.;
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++)
            args[d] = ippl::Index(nPoints[d]);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        ippl::e_dim_tag allParallel[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < Dim; d++) {
            allParallel[d] = ippl::PARALLEL;
            hx[d]          = domain[d] / nPoints[d];
            origin[d]      = 0;
        }

        const bool isAllPeriodic = true;
        auto& layout             = std::get<Idx>(layouts) =
            flayout_type<Dim>(owned, allParallel, isAllPeriodic);

        auto& mesh = std::get<Idx>(meshes) = mesh_type<Dim>(owned, hx, origin);

        auto field = std::get<Idx>(fields) = std::make_shared<field_type<Dim>>(mesh, layout);

        auto& pl = std::get<Idx>(playouts) = playout_type<Dim>(layout, mesh);

        auto bunch = std::get<Idx>(bunches) = std::make_shared<bunch_type<Dim>>(pl);

        int nRanks = ippl::Comm->size();
        if (nParticles % nRanks > 0) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
            exit(1);
        }

        size_t nloc = nParticles / nRanks;
        bunch->create(nloc);

        std::mt19937_64 eng;
        eng.seed(42);
        eng.discard(nloc * ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(0, 1);

        auto R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < nloc; ++i) {
            for (unsigned d = 0; d < Dim; d++) {
                R_host(i)[d] = unif(eng) * domain[d];
            }
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);

        std::get<Idx>(orbs).initialize(layout, mesh, *field);
    }

    template <unsigned Dim>
    void repartition() {
        bool fromAnalyticDensity = false;

        constexpr unsigned Idx = dimToIndex(Dim);

        auto bunch   = std::get<Idx>(bunches);
        auto field   = std::get<Idx>(fields);
        auto& orb    = std::get<Idx>(orbs);
        auto& layout = std::get<Idx>(layouts);
        auto& mesh   = std::get<Idx>(meshes);

        orb.binaryRepartition(bunch->R, layout, fromAnalyticDensity);
        field->updateLayout(layout);
        bunch->updateLayout(layout, mesh);
    }

    template <unsigned Dim>
    ippl::NDIndex<Dim> getDomain() {
        return std::get<dimToIndex(Dim)>(layouts).getDomain();
    }

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, bunch_type> bunches;
    size_t nParticles;
    size_t nPoints[MaxDim];
    double domain[MaxDim];

    Collection<flayout_type> layouts;
    Collection<mesh_type> meshes;
    Collection<playout_type> playouts;
    Collection<ORB> orbs;
};

TEST_F(ORBTest, Volume) {
    auto check = [&]<unsigned Dim>(playout_type<Dim>& pl, std::shared_ptr<bunch_type<Dim>>& bunch) {
        ippl::NDIndex<Dim> dom = getDomain<Dim>();
        bunch_type<Dim> buffer(pl);

        pl.update(*bunch, buffer);

        repartition<Dim>();

        pl.update(*bunch, buffer);

        ippl::NDIndex<Dim> ndom = getDomain<Dim>();

        ASSERT_DOUBLE_EQ(dom.size(), ndom.size());
    };

    apply(check, playouts, bunches);
}

TEST_F(ORBTest, Charge) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field,
                                   std::shared_ptr<bunch_type<Dim>>& bunch, playout_type<Dim>& pl) {
        bunch_type<Dim> buffer(pl);

        double charge = 0.5;

        bunch->Q = charge;

        pl.update(*bunch, buffer);

        repartition<Dim>();

        pl.update(*bunch, buffer);

        *field = 0.0;
        scatter(bunch->Q, *field, bunch->R);

        double totalCharge = field->sum();

        ASSERT_NEAR((nParticles * charge - totalCharge) / totalCharge, 0., 1e-13);
    };

    apply(check, fields, bunches, playouts);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    { ::testing::InitGoogleTest(&argc, argv); }
    ippl::finalize();
    return RUN_ALL_TESTS();
}
