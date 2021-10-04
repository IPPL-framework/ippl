// Uniform Plasma Test
//
//   Usage:
//     srun ./UniformPlasmaTest 128 128 128 10000 10 FFT --info 10
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
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
#include "ChargedParticles.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <set>
#include <chrono>

#include<Kokkos_Random.hpp>

#include <random>
#include "Utility/IpplTimings.h"

const char* TestName = "UniformPlasmaTest";

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  // Output View for the random numbers
  view_type vals;

  // The GeneratorPool
  GeneratorPool rand_pool;

  T start, end;

  // Initialize all members
  generate_random(view_type vals_, GeneratorPool rand_pool_, 
                  T start_, T end_)
      : vals(vals_), rand_pool(rand_pool_), 
        start(start_), end(end_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    // Draw samples numbers from the pool as double in the range [start, end)
    for (unsigned d = 0; d < Dim; ++d) {
      vals(i)[d] = rand_gen.drand(start[d], end[d]);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg("UniformPlasmaTest");
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Ippl::Comm->setDefaultOverallocation(2);

    auto start = std::chrono::high_resolution_clock::now();
    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef FirstUpdateTimer = IpplTimings::getTimer("initialisation");
    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("kick");
    static IpplTimings::TimerRef temp = IpplTimings::getTimer("randomMove");
    static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("drift");
    static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
    static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
    static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("domainDecomp");

    IpplTimings::startTimer(mainTimer);

    const size_type totalP = std::atoll(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);

    msg << "Uniform Plasma Test"
        << endl
        << "nt " << nt << " Np= "
        << totalP << " grid = " << nr
        << endl;

    using bunch_type = ChargedParticles<PLayout_t>;

    std::unique_ptr<bunch_type>  P;

    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }

    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::PARALLEL;
    }

    // create mesh and layout objects for this problem domain
    Vector_t rmin(0.0);
    Vector_t rmax(20.0);
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};
    const double dt = 1.0;

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    double Q = -1562.5;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);

    P->nr_m = nr;
    size_type nloc = totalP / Ippl::Comm->size();

    int rest = (int) (totalP - nloc * Ippl::Comm->size());

    if ( Ippl::Comm->rank() < rest )
        ++nloc;

    IpplTimings::startTimer(particleCreation);
    P->create(nloc);

    const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
    Vector_t Rmin, Rmax;
    for (unsigned d = 0; d <Dim; ++d) {
        Rmin[d] = origin[d] + lDom[d].first() * hr[d];
        Rmax[d] = origin[d] + (lDom[d].last() + 1) * hr[d];
    }

    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*Ippl::Comm->rank()));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         P->R.getView(), rand_pool64, Rmin, Rmax));
    Kokkos::fence();
    P->q = P->Q_m/totalP;
    P->P = 0.0;
    IpplTimings::stopTimer(particleCreation);

    IpplTimings::startTimer(FirstUpdateTimer);
    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);
    P->initializeORB(FL, mesh);

    bunch_type bunchBuffer(PL);

	IpplTimings::startTimer(updateTimer);
    PL.update(*P, bunchBuffer);
    IpplTimings::stopTimer(updateTimer);

    msg << "particles created and initial conditions assigned " << endl;

    P->stype_m = argv[6];
    P->initSolver();
    P->time_m = 0.0;
    P->loadbalancefreq_m = std::atoi(argv[7]);

    unsigned int nstep = 0;
    if (P->balance(totalP, nstep)) {
        msg << "Starting first repartition" << endl;
        IpplTimings::startTimer(domainDecomposition);
        P->repartition(FL, mesh, bunchBuffer);
        IpplTimings::stopTimer(domainDecomposition);
    }
    
    P->scatterCIC(totalP, 0, hr);

    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpData();
    IpplTimings::stopTimer(dumpDataTimer);

    IpplTimings::stopTimer(FirstUpdateTimer);

    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    //P->gatherStatistics(totalP);
    for (unsigned int it=0; it<nt; it++) {

        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        // kick

        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E * 0.0;
        IpplTimings::stopTimer(PTimer);

        IpplTimings::startTimer(temp);
        Kokkos::parallel_for(P->getLocalNum(),
                             generate_random<Vector_t, 
                             Kokkos::Random_XorShift64_Pool<>, Dim>(
                             P->P.getView(), rand_pool64, -hr, hr));
        Kokkos::fence();
        IpplTimings::stopTimer(temp);

        //drift
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);

        //Since the particles have moved spatially update them to correct processors
	    IpplTimings::startTimer(updateTimer);
        PL.update(*P, bunchBuffer);
        IpplTimings::stopTimer(updateTimer);


        // Domain Decomposition
        if (P->balance(totalP, it+1)) {
           msg << "Starting repartition" << endl;
           IpplTimings::startTimer(domainDecomposition);
           P->repartition(FL, mesh, bunchBuffer);
           IpplTimings::stopTimer(domainDecomposition);
        }

        //scatter the charge onto the underlying grid
        P->scatterCIC(totalP, it+1, hr);

        //Field solve
        IpplTimings::startTimer(SolveTimer);
        P->solver_mp->solve();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        P->gatherCIC();

        //kick
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E * 0.0;
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);
        P->dumpData();
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it+1 << " time: " << P->time_m << endl;
        //P->gatherStatistics(totalP);
    }

    msg << "Uniform Plasma Test: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
