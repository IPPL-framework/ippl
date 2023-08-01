// Bump on Tail Instability/Two-stream Instability Test
//   Usage:
//     srun ./BumponTailInstability
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./BumponTailInstability 128 128 128 10000 10 FFT 0.01 --overallocate 2.0 --info 10
//     Change the TestName to TwoStreamInstability or BumponTailInstability
//     in order to simulate the Two stream instability or bump on tail instability
//     cases
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

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"
#include "Distribution/Distribution.hpp"

constexpr unsigned Dim = 3;

const char* TestName = "BumponTailInstability";

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        setSignalHandler();

        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);

        int arg = 1;

        auto start = std::chrono::high_resolution_clock::now();
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        };

        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        static IpplTimings::TimerRef dumpDataTimer    = IpplTimings::getTimer("dumpData");
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");

        IpplTimings::startTimer(mainTimer);

        const size_type totalP = std::atoll(argv[arg++]);
        const unsigned int nt  = std::atoi(argv[arg++]);

        msg << TestName << endl << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

        using bunch_type = ChargedParticles<PLayout_t<double, Dim>, double, Dim>;

        std::shared_ptr<bunch_type> P;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        double kw = 0.0;  // FixMe: this is ALSO defined in the distribution.
                          // need this here for rmax
        if (std::strcmp(TestName, "TwoStreamInstability") == 0) {
            // Parameters for two stream instability as in
            //  https://www.frontiersin.org/articles/10.3389/fphy.2018.00105/full
            kw = 0.5;
        } else if (std::strcmp(TestName, "BumponTailInstability") == 0) {
            kw = 0.21;
        } else {
            // Default value is two stream instability
            kw = 0.5;
        }

        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax = 2 * pi / kw;

        Vector_t<double, Dim> hr;
        for (unsigned d = 0; d < Dim; d++) {
            hr[d] = rmax[d] / nr[d];
        }
        Vector_t<double, Dim> origin = rmin;
        const double dt              = std::min(.05, 0.5 * *std::min_element(hr.begin(), hr.end()));

        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(domain, decomp, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        // Q = -\int\int f dx dv
        double Q           = std::reduce(rmax.begin(), rmax.end(), -1., std::multiplies<double>());
        std::string solver = argv[arg++];

        if (solver == "OPEN") {
            throw IpplException("BumpOnTailInstability",
                                "Open boundaries solver incompatible with this simulation!");
        }

        P = std::make_shared<bunch_type>(PL, hr, rmin, rmax, decomp, Q, solver);

        P->nr_m = nr;

        P->initializeFields(mesh, FL);

        bunch_type bunchBuffer(PL);

        P->initSolver();
        P->time_m                 = 0.0;
        P->loadbalancethreshold_m = std::atof(argv[arg++]);

        Distribution::BumponTailDistribution<double, 3> dist(rmax, rmin, hr, origin, totalP,
                                                             TestName);

        bool isFirstRepartition = false;

        if ((P->loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            msg << "Starting first repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
            dist.repartitionRhs(P.get(), lDom);
            P->initializeORB(FL, mesh);
            P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
            IpplTimings::stopTimer(domainDecomposition);
        }

        msg << "First domain decomposition done" << endl;

        IpplTimings::startTimer(particleCreation);
        dist.createParticles(P.get(), PL.getRegionLayout());
        IpplTimings::stopTimer(particleCreation);

        typename ParticleAttrib<Vector_t<double, Dim>>::HostMirror RHost = P->R.getHostMirror();
        typename ParticleAttrib<Vector_t<double, Dim>>::HostMirror VHost = P->V.getHostMirror();
        Kokkos::deep_copy(RHost, P->R.getView());
        Kokkos::deep_copy(VHost, P->V.getView());

        Connector::StatisticsConnector<double, Dim> statConn(TestName, totalP);

        P->q = P->Qtot_m / totalP;
        msg << "particles created and initial conditions assigned " << endl;
        isFirstRepartition = false;
        // The update after the particle creation is not needed as the
        // particles are generated locally

        IpplTimings::startTimer(DummySolveTimer);
        P->rhs_m = 0.0;
        P->runSolver();
        IpplTimings::stopTimer(DummySolveTimer);

        P->scatterCIC(totalP, 0, hr);

        IpplTimings::startTimer(SolveTimer);
        P->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        P->gatherCIC();

        IpplTimings::startTimer(dumpDataTimer);
        statConn.gatherLoadBalancingStatistics(P->getLocalNum(), P->time_m);
        statConn.gatherLocalDomainStatistics(FL, 0);
        statConn.gatherFieldStatistics(P->V, P->rhs_m, P->F_m, hr, P->getLocalNum(), P->time_m);
        statConn.dumpBumponTail(P->F_m, hr, P->time_m);
        IpplTimings::stopTimer(dumpDataTimer);

        // begin main timestep loop
        msg << "Starting iterations ..." << endl;
        for (unsigned int it = 0; it < nt; it++) {
            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick

            IpplTimings::startTimer(PTimer);
            P->V = P->V - 0.5 * dt * P->E;
            IpplTimings::stopTimer(PTimer);

            // drift
            IpplTimings::startTimer(RTimer);
            P->R = P->R + dt * P->V;
            IpplTimings::stopTimer(RTimer);

            // Since the particles have moved spatially update them to correct processors
            IpplTimings::startTimer(updateTimer);
            PL.update(*P, bunchBuffer);
            IpplTimings::stopTimer(updateTimer);

            // Domain Decomposition
            if (P->balance(totalP, it + 1, TestName)) {
                msg << "Starting repartition" << endl;
                IpplTimings::startTimer(domainDecomposition);
                P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);

                IpplTimings::startTimer(dumpDataTimer);
                statConn.gatherLocalDomainStatistics(FL, it + 1);
                IpplTimings::stopTimer(dumpDataTimer);
            }

            // scatter the charge onto the underlying grid
            P->scatterCIC(totalP, it + 1, hr);

            // Field solve
            IpplTimings::startTimer(SolveTimer);
            P->runSolver();
            IpplTimings::stopTimer(SolveTimer);

            // gather E field
            P->gatherCIC();

            // kick
            IpplTimings::startTimer(PTimer);
            P->V = P->V - 0.5 * dt * P->E;
            IpplTimings::stopTimer(PTimer);

            P->time_m += dt;
            IpplTimings::startTimer(dumpDataTimer);
            statConn.dumpBumponTail(P->F_m, hr, P->time_m);
            statConn.gatherLoadBalancingStatistics(P->getLocalNum(), P->time_m);
            statConn.gatherLocalDomainStatistics(FL, it + 1);
            statConn.gatherFieldStatistics(P->V, P->rhs_m, P->F_m, hr, P->getLocalNum(), P->time_m);
            IpplTimings::stopTimer(dumpDataTimer);
            msg << "Finished time step: " << it + 1 << " time: " << P->time_m << endl;

            if (checkSignalHandler()) {
                msg << "Aborting timestepping loop due to signal " << interruptSignalReceived
                    << endl;
                break;
            }
        }

        msg << TestName << ": End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_chrono =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Elapsed time: " << time_chrono.count() << std::endl;
    }
    ippl::finalize();

    return 0;
}
