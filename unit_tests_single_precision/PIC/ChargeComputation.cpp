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

class ORBComp{

public:
    static constexpr size_t dim = 3;
    typedef ippl::FieldLayout<dim> flayout_type;
    typedef ippl::UniformCartesian<float, dim> mesh_type;
    typedef ippl::Field<float, dim, mesh_type> field_type;
    typedef ippl::ParticleSpatialLayout<float, dim> playout_type;
    typedef ippl::OrthogonalRecursiveBisection<float, dim, mesh_type> ORB;

    template<class PLayout>
    struct Bunch : public ippl::ParticleBase<PLayout>
    {
        Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout)
        {
            this->addAttribute(Q);
        }
        
        ~Bunch(){ }
        
        typedef ippl::ParticleAttrib<float> charge_container_type;
        charge_container_type Q;

        void updateLayout(flayout_type fl, mesh_type mesh) {
            PLayout& layout = this->getLayout();
            layout.updateLayout(fl, mesh);
        }
    };

    typedef Bunch<playout_type> bunch_type;

    ORBComp(size_t nParticles_t, size_t  nPoints_t)
    // Original configuration 256^3 particles, 512^3 grid.
    : nParticles{nParticles_t}
    , nPoints{nPoints_t}
    {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = ippl::PARALLEL;

        const bool isAllPeriodic = true;
        layout_m = flayout_type(owned, allParallel, isAllPeriodic);

        float dx = 1.0 / float(nPoints);
        ippl::Vector<float, dim> hx = {dx, dx, dx};
        ippl::Vector<float, dim> origin = {0, 0, 0};

        mesh_m = mesh_type(owned, hx, origin);
      
        field = std::make_unique<field_type>(mesh_m, layout_m);
      
        pl_m = playout_type(layout_m, mesh_m);
             
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
        std::uniform_real_distribution<float> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = bunch->R.getHostMirror();
        for(size_t i = 0; i < nloc; ++i) {
            for (int d = 0; d < 3; d++) {
                R_host(i)[d] = unif(eng);
            }
        }

        Kokkos::deep_copy(bunch->R.getView(), R_host);

        orb.initialize(layout_m, mesh_m, *field);
 
        
    }


    void repartition() {
        bool fromAnalyticDensity = false;
        orb.binaryRepartition(bunch->R, layout_m, 
                              fromAnalyticDensity);
 
        field->updateLayout(layout_m);
 
        bunch->updateLayout(layout_m, mesh_m);
 
        
    }

    ippl::NDIndex<dim> getDomain() {
        return layout_m.getDomain();
    }

    std::unique_ptr<field_type> field;
    std::unique_ptr<bunch_type> bunch;
    size_t nParticles;
    size_t nPoints;

    flayout_type layout_m;
    mesh_type mesh_m;
    playout_type pl_m;
    ORB orb;

float error() {

    bunch_type buffer(pl_m);

    float charge = 0.5;

    bunch->Q = charge;

    pl_m.update(*bunch, buffer);

    repartition();
    pl_m.update(*bunch, buffer);

    *field = 0.0;
    scatter(bunch->Q, *field, bunch->R);

    float totalCharge = field->sum();

    return (nParticles * charge - totalCharge) / totalCharge;
}

};

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

    size_t nParticles = atoi( argv[1] );
    size_t nPoints = atoi( argv[2] );
    
    ORBComp orb_t(nParticles, nPoints);
    float error = orb_t.error();
    if( Ippl::Comm->rank() == 0 ) 
         std::cout <<  nPoints << "^3 " << error  << std::endl;
    
    return 0;
}
