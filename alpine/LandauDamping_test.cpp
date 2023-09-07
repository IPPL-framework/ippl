// Landau Damping Test
//   Usage:
//     srun ./LandauDamping
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
//     srun ./LandauDamping 128 128 128 10000 10 FFT 0.01 --overallocate 2.0 --info 10
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

#include "Ippl.h"

#include "Utility/IpplTimings.h"

//#include "ChargedParticles.hpp"
#include "Manager/PicManager.h"
#include "datatypes.h"
#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
 
 #include "Random/InverseTransformSampling.h"    
 
constexpr unsigned Dim = 3;
using T = double;

KOKKOS_FUNCTION
double CDF(double y, double alpha, double k) {
    return y + (alpha / k) * std::sin(k * y);
}

KOKKOS_FUNCTION
double PDF(double y, double alpha, double k) {
    return  (1.0 + alpha * Kokkos::cos(k * y));
}

KOKKOS_FUNCTION
double ESTIMATE(double u, double alpha) {
    return u/(1+alpha); // maybe E[x] is good enough as the first guess
}


const char* TestName = "LandauDamping";

//template <typename T, unsigned Dim>
class MyPicManager : public ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>> {
public:
    double time_m;
    double Q_m;
    MyPicManager()
        : ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>>(){
    }

    void par2grid() override {
        scatterCIC();
    }

    void grid2par() override {
        gatherCIC();
    }
    
    void gatherCIC() { gather(pcontainer_m->E, fcontainer_m->E_m, pcontainer_m->R); }
    
    void scatterCIC() {
        Inform m("scatter ");

        fcontainer_m->rho_m = 0.0;
        scatter(pcontainer_m->q, fcontainer_m->rho_m, pcontainer_m->R);

         std::cout << std::fabs((Q_m - fcontainer_m->rho_m.sum()) / Q_m)  << std::endl;

        size_type Total_particles = 0;
        size_type local_particles = pcontainer_m->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        double cellVolume =
            std::reduce(fcontainer_m->hr_m.begin(), fcontainer_m->hr_m.end(), 1., std::multiplies<double>());
        fcontainer_m->rho_m = fcontainer_m->rho_m / cellVolume;

        // rho = rho_e - rho_i (only if periodic BCs)
        if (fsolver_m->stype_m != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= fcontainer_m->rmax_m[d] - fcontainer_m->rmin_m[d];
            }
            fcontainer_m->rho_m = fcontainer_m->rho_m - (Q_m / size);
        }
    }
    
    void dumpLandau() { dumpLandau(fcontainer_m->E_m.getView()); }

    template <typename View>
    void dumpLandau(const View& Eview) {
        const int nghostE = fcontainer_m->E_m.getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Eview, nghostE),
            KOKKOS_LAMBDA(const index_array_type& args, double& E2, double& ENorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Eview, args)[0];
                double e2  = Kokkos::pow(val, 2);
                E2 += e2;

                double norm = Kokkos::fabs(ippl::apply(Eview, args)[0]);
                if (norm > ENorm) {
                    ENorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        MPI_Reduce(&localEx2, &globaltemp, 1, MPI_DOUBLE, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());
        double fieldEnergy =
            std::reduce(fcontainer_m->hr_m.begin(), fcontainer_m->hr_m.end(), globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        MPI_Reduce(&localExNorm, &ExAmp, 1, MPI_DOUBLE, MPI_MAX, 0, ippl::Comm->getCommunicator());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldLandau_";
            fname << ippl::Comm->size();
            fname<<"_test";
            fname << ".csv";

            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);

            if (time_m == 0.0) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }

            csvout << time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }

        ippl::Comm->barrier();
    }
    
};


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("LandauDamping");
        Inform msg2all("LandauDamping", INFORM_ALL_NODES);

        int arg = 1;

        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        const size_type totalP = std::atoll(argv[arg++]);
        const unsigned int nt  = std::atoi(argv[arg++]);

        msg << "Landau damping" << endl
            << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

       ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = ippl::PARALLEL;
        }
        
        // create mesh and layout objects for this problem domain
        Vector_t<double, Dim> kw = 0.5;
        double alpha             = 0.05;
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax = 2 * pi / kw;

        Vector_t<double, Dim> hr = rmax / nr;
        // Q = -\int\int f dx dv
        double Q = std::reduce(rmax.begin(), rmax.end(), -1., std::multiplies<double>());
        Vector_t<double, Dim> origin = rmin;
        const double dt              = std::min(.05, 0.5 * *std::min_element(hr.begin(), hr.end()));
        
        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(domain, decomp, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        std::string solver = argv[arg++];

        if (solver == "OPEN") {
            throw IpplException("LandauDamping",
                                "Open boundaries solver incompatible with this simulation!");
        }
        
        using ParticleContainer_t = ParticleContainer<T, Dim>;
        std::shared_ptr<ParticleContainer_t> pc = std::make_shared<ParticleContainer_t>(PL);
        
        using FieldContainerType = FieldContainer<T, Dim>;
        std::shared_ptr<FieldContainerType> fc = std::make_shared<FieldContainerType>(hr, rmin, rmax, decomp);
        
        fc->initializeFields(mesh, FL);
        
        using FieldSolverType = FieldSolver<T, Dim>;
        std::shared_ptr<FieldSolverType> fs = std::make_shared<FieldSolverType>(solver, fc->rho_m, fc->E_m);
        
        fs->initSolver();
                
        MyPicManager manager;
        manager.Q_m = Q;
        manager.setParticleContainer(pc);
        manager.setFieldContainer(fc);
        manager.setFieldSolver(fs);

         // Sample particle positions:
         ippl::detail::RegionLayout<double, 3, Mesh_t<3>> rlayout(FL, mesh);
         using InvTransSampl_t = ippl::random::InverseTransformSampling<double, 3, Kokkos::DefaultExecutionSpace>;
         ippl::random::Distribution<double, 3> distR;
         for(int d=0; d<3; d++){
                 double k = kw[d];
		 distR.setCdfFunction(d, [alpha, k](double y) { return CDF(y, alpha, k);});
		 distR.setPdfFunction(d, [alpha, k](double y) { return PDF(y, alpha, k);});
		 distR.setEstimationFunction(d, [alpha](double u) { return ESTIMATE(u, alpha);});
         }
         InvTransSampl_t its(rmin, rmax, rlayout, distR, totalP);
         unsigned int nloc = its.getLocalNum();
         pc->create(nloc);
         its.generate(distR, pc->R.getView(), 42 + 100 * ippl::Comm->rank());
         
         // Sample particle velocity:
         // Box-Muller method
         //Kokkos::parallel_for(
         //   nloc, ippl::random::generate_random_normal<double, Kokkos::DefaultExecutionSpace, 3>(
         //            pc->P.getView(), 0.0, 1.0, 42 + 100 * ippl::Comm->rank()));
         
         // standard method
         Kokkos::Random_XorShift64_Pool<> rand_pool64_((size_type)(42 + 100 * ippl::Comm->rank()));
         Kokkos::parallel_for(
            nloc, ippl::random::generate_random_normal_basic<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                     pc->P.getView(), rand_pool64_));
        
        Kokkos::fence();
        ippl::Comm->barrier();

        pc->q = Q / totalP;
        msg << "particles created and initial conditions assigned " << endl;
        
        fc->rho_m = 0.0;
        fs->runSolver();
        manager.par2grid();
        fs->runSolver();
        manager.grid2par();

        manager.time_m = 0.0;

        // begin main timestep loop
        msg << "Starting iterations ..." << endl;
        for (unsigned int it = 0; it < nt; it++) {
            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick

            pc->P = pc->P - 0.5 * dt * pc->E;

            // drift
            pc->R = pc->R + dt * pc->P;

            // Since the particles have moved spatially update them to correct processors
            pc->update();

            // Domain Decomposition
            //if (P->balance(totalP, it + 1)) {
            //    msg << "Starting repartition" << endl;
            //    IpplTimings::startTimer(domainDecomposition);
            //    P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
            //    IpplTimings::stopTimer(domainDecomposition);
            //    // IpplTimings::startTimer(dumpDataTimer);
            //    // P->dumpLocalDomains(FL, it+1);
            //    // IpplTimings::stopTimer(dumpDataTimer);
            //}

            // scatter the charge onto the underlying grid
            manager.par2grid();
            
            // Field solve
            fs->runSolver();

            // gather E field
            manager.grid2par();

            // kick
            pc->P = pc->P - 0.5 * dt * pc->E;


            manager.time_m += dt;            
            manager.dumpLandau();

            msg << "Finished time step: " << it + 1 << " time: " << manager.time_m << endl;
        }
        msg << "LandauDamping: End." << endl;
        
    }
    ippl::finalize();

    return 0;
}
