// Penning Trap
//
//   Usage:
//     srun ./PenningTrap 128 128 128 10000 300 FFT Gaussian 1.0 --info 10
//     srun ./PenningTrap 128 128 128 10000 300 FFT Uniform 1.0 --info 10
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

#include <random>
#include <cmath>
#include "Utility/IpplTimings.h"

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  // Output View for the random numbers
  view_type x, v;

  // The GeneratorPool
  GeneratorPool rand_pool;

  T mu, sd, startU1, endU1, startU2, endU2;

  double pi = acos(-1.0);

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, 
                  T& mu_, T& sd_, T& startU1_, T& endU1_, T& startU2_, 
                  T& endU2_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        mu(mu_), sd(sd_), startU1(startU1_), endU1(endU1_),
        startU2(startU2_), endU2(endU2_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u1, u2;
    for (unsigned d = 0; d < Dim; ++d) {
        
        u1 = rand_gen.drand(startU1[d], endU1[d]);
        u2 = rand_gen.drand(startU2[d], endU2[d]);
        x(i)[d] = sd[d] * (std::sqrt(-2.0 * std::log(u1)) *
                           std::cos(2.0 * pi * u2)) + mu[d];
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(double& x, double& mu, double& sd) {
   double cdf = 0.5 * std::erf((x - mu)/(sd * std::sqrt(2)));
   return cdf;
}

const char* TestName = "PenningTrap";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg("PenningTrap");
    Inform msg2all(argv[0],INFORM_ALL_NODES);

    Ippl::Comm->setDefaultOverallocation(2.0);


    auto start = std::chrono::high_resolution_clock::now();
    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    IpplTimings::startTimer(mainTimer);
    const size_type totalP = std::atol(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);

    msg << "Penning Trap "
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
    const double dt = 0.95*dx;//size of timestep

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    double Q = -1562.5;
    double Bext = 5.0;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);

    P->nr_m = nr;

    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    IpplTimings::startTimer(particleCreation);

    Vector_t length = rmax - rmin;

    std::vector<double> mu(2*Dim);
    std::vector<double> sd(2*Dim);
    std::vector<double> states(2*Dim);


    for (unsigned d = 0; d<Dim; d++) {
        mu[d] = length[d]/2;
        mu[Dim + d] = 0.0;
        sd[Dim + d] = 1.0;
    }
    sd[0] = 0.15*length[0];
    sd[1] = 0.05*length[1];
    sd[2] = 0.20*length[2];

    const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
    Vector_t Rmin, Rmax, Nr, Dr, startU1, endU1, startU2, endU2;
    for (unsigned d = 0; d <Dim; ++d) {
        Rmin[d] = origin[d] + lDom[d].first() * hr[d];
        Rmax[d] = origin[d] + (lDom[d].last() + 1) * hr[d];
        Nr[d] = CDF(Rmax[d], mu[d], sd[d]) - CDF(Rmin[d], mu[d], sd[d]);  
        Dr[d] = CDF(rmax[d], mu[d], sd[d]) - CDF(rmin[d], mu[d], sd[d]);
        startU1[d] = 0.01;
        endU1[d] = 0.99;
        start[d] = (1.0 / 2 * pi) * std::acos(-std::pow(Rmin[d],2)/
                    (2 * std::log(startU1[d]))); 
        end[d] = (1.0 / 2 * pi) * std::acos(-std::pow(Rmax[d],2)/
                    (2 * std::log(endU1[d]))); 
    }

    double factor = (Nr[0] * Nr[1] * Nr[2]) / (Dr[0] * Dr[1] * Dr[2]);
    size_type nloc = (size_type)(totalP * factor);
    size_type Total_particles = 0;

    MPI_Allreduce(&nloc, &Total_particles, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, Ippl::getComm());

    int rest = (int) (totalP - Total_particles);

    if ( Ippl::Comm->rank() < rest )
        ++nloc;

    P->create(nloc);
    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*Ippl::Comm->rank()));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         P->R.getView(), P->P.getView(), rand_pool64, mu, sd, startU1, endU1, 
                         startU2, endU2));
    Kokkos::fence();
    Ippl::Comm->barrier();
    //std::mt19937_64 eng[4*Dim];
    //for (unsigned i = 0; i < 4*Dim; ++i) {
    //    eng[i].seed(42 + i * Dim);
    //    eng[i].discard( nloc * Ippl::Comm->rank());
    //}


    //std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);

    //typename bunch_type::particle_position_type::HostMirror R_host = P->R.getHostMirror();
    //typename bunch_type::particle_position_type::HostMirror P_host = P->P.getHostMirror();

    //double sum_coord=0.0;
    //for (size_type i = 0; i< nloc; i++) {
    //    for (unsigned istate = 0; istate < 2*Dim; ++istate) {
    //        double u1 = dist_uniform(eng[istate*2]);
    //        double u2 = dist_uniform(eng[istate*2+1]);
    //        states[istate] = sd[istate] * (std::sqrt(-2.0 * std::log(u1)) *
    //                         std::cos(2.0 * pi * u2)) + mu[istate];
    //    }
    //    for (unsigned d = 0; d<Dim; d++) {
    //        R_host(i)[d] =  std::fabs(std::fmod(states[d],length[d]));
    //        sum_coord += R_host(i)[d];
    //        P_host(i)[d] = states[Dim + d];
    //    }
    //}
    /////Just to check are we getting the same particle distribution for
    ////different no. of processors
    //double global_sum_coord = 0.0;
    //MPI_Reduce(&sum_coord, &global_sum_coord, 1,
    //           MPI_DOUBLE, MPI_SUM, 0, Ippl::getComm());

    //if(Ippl::Comm->rank() == 0) {
    //    std::cout << "Sum Coord: " << std::setprecision(16) << global_sum_coord << std::endl;
    //}

    //Kokkos::deep_copy(P->R.getView(), R_host);
    //Kokkos::deep_copy(P->P.getView(), P_host);

    P->q = P->Q_m/totalP;

    IpplTimings::stopTimer(particleCreation);                                                    
    
    
    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);
    P->initializeORB(FL, mesh);

    bunch_type bunchBuffer(PL);
    static IpplTimings::TimerRef FirstUpdateTimer = IpplTimings::getTimer("FirstUpdate");           
    IpplTimings::startTimer(FirstUpdateTimer);                                               
    PL.update(*P, bunchBuffer);
    IpplTimings::stopTimer(FirstUpdateTimer);

    msg << "particles created and initial conditions assigned " << endl;

    P->stype_m = argv[6];
    P->initSolver();
    P->time_m = 0.0;
    P->loadbalancethreshold_m = std::atof(argv[8]);

    static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("domainDecomp");
    unsigned int nstep = 0;
    if (P->balance(totalP, nstep)) {
        msg << "Starting first repartition" << endl;
        IpplTimings::startTimer(domainDecomposition);
        P->repartition(FL, mesh, bunchBuffer);
        IpplTimings::stopTimer(domainDecomposition);
    }

    P->scatterCIC(totalP, 0, hr);

    static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("Solve");
    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();

    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
    IpplTimings::startTimer(dumpDataTimer);
    P->dumpData();
    P->gatherStatistics(totalP);
    IpplTimings::stopTimer(dumpDataTimer);

    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it=0; it<nt; it++) {

        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        // kick
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityPush");
        IpplTimings::startTimer(PTimer);
        auto Rview = P->R.getView();
        auto Pview = P->P.getView();
        auto Eview = P->E.getView();
        double V0 = 30*rmax[2];
        Kokkos::parallel_for("Kick1", P->getLocalNum(),
                              KOKKOS_LAMBDA(const size_t j){
            double Eext_x = -(Rview(j)[0] - (rmax[0]/2)) * (V0/(2*pow(rmax[2],2)));
            double Eext_y = -(Rview(j)[1] - (rmax[1]/2)) * (V0/(2*pow(rmax[2],2)));
            double Eext_z =  (Rview(j)[2] - (rmax[2]/2)) * (V0/(pow(rmax[2],2)));

            Pview(j)[0] -= 0.5 * dt * ((Eview(j)[0] + Eext_x) + Pview(j)[1] * Bext);
            Pview(j)[1] -= 0.5 * dt * ((Eview(j)[1] + Eext_y) - Pview(j)[0] * Bext);
            Pview(j)[2] -= 0.5 * dt *  (Eview(j)[2] + Eext_z);
        });
        IpplTimings::stopTimer(PTimer);

        //drift
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionPush");
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P;
        Ippl::Comm->barrier();
        IpplTimings::stopTimer(RTimer);

        //Since the particles have moved spatially update them to correct processors
        PL.update(*P, bunchBuffer);

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
        auto R2view = P->R.getView();
        auto P2view = P->P.getView();
        auto E2view = P->E.getView();
        Kokkos::parallel_for("Kick2", P->getLocalNum(),
                              KOKKOS_LAMBDA(const size_t j){
            double Eext_x = -(R2view(j)[0] - (rmax[0]/2)) * (V0/(2*pow(rmax[2],2)));
            double Eext_y = -(R2view(j)[1] - (rmax[1]/2)) * (V0/(2*pow(rmax[2],2)));
            double Eext_z =  (R2view(j)[2] - (rmax[2]/2)) * (V0/(pow(rmax[2],2)));

            P2view(j)[0] -= 0.5 * dt * ((E2view(j)[0] + Eext_x) + P2view(j)[1] * Bext);
            P2view(j)[1] -= 0.5 * dt * ((E2view(j)[1] + Eext_y) - P2view(j)[0] * Bext);
            P2view(j)[2] -= 0.5 * dt *  (E2view(j)[2] + Eext_z);
        });
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);
        P->dumpData();
        P->gatherStatistics(totalP);
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it+1 << " time: " << P->time_m << endl;
    }

    msg << "Penning Trap: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
