// Penning Trap
//
//   Usage:
//     srun ./PenningTrap 128 128 128 10000 300 FFT 1.0 --info 10
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
#include <cmath>
#include <set>
#include <chrono>

#include<Kokkos_Random.hpp>
#include <random>
#include "Utility/IpplTimings.h"

template <typename T>
struct Newton1D {

  double tol = 1e-12;
  int max_iter = 20;
  double pi = std::acos(-1.0);
  
  T mu, sigma, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& mu_, const T& sigma_, 
           const T& u_) 
  : mu(mu_), sigma(sigma_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = std::erf((x - mu)/(sigma * std::sqrt(2.0))) 
          - 2 * u + 1;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = (1 / sigma) * std::sqrt(2 / pi) * 
               std::exp(-0.5 * (std::pow(((x - mu) / sigma),2)));
      return Fprime;
  }

  KOKKOS_FUNCTION
  void solve(T& x) {
      int iterations = 0;
      while ((iterations < max_iter) && (std::fabs(f(x)) > tol)) {
          x = x - (f(x)/fprime(x));
          iterations += 1;
      }
  }
};


template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  // Output View for the random numbers
  view_type x, v;

  // The GeneratorPool
  GeneratorPool rand_pool;

  T mu, sigma, minU, maxU;

  double pi = std::acos(-1.0);

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_,
                  T& mu_, T& sigma_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        mu(mu_), sigma(sigma_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u;
    for (unsigned d = 0; d < Dim; ++d) {
        u = rand_gen.drand(minU[d], maxU[d]);
        x(i)[d] = (std::sqrt(pi / 2) * (2 * u - 1)) * 
                  sigma[d] + mu[d];
        Newton1D<value_type> solver(mu[d], sigma[d], u);
        solver.solve(x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, double& mu, double& sigma) {
   double cdf = 0.5 * (1.0 + std::erf((x - mu)/(sigma * std::sqrt(2))));
   return cdf;
}


KOKKOS_FUNCTION
Vector_t PDF(const Vector_t& xvec, const Vector_t&mu, 
             const Vector_t& sigma, const unsigned Dim) {
    Vector_t pdf = 1.0;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf[d] *= (1.0/ (sd[0] * std::sqrt(2 * M_PI))) * 
                  std::exp(-0.5 * std::pow((xvec[d] - mu[d])/sigma[d],2));
    }
    return pdf;
}

const char* TestName = "PenningTrap";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    Inform msg("PenningTrap");
    Inform msg2all("PenningTrap",INFORM_ALL_NODES);


    Ippl::Comm->setDefaultOverallocation(std::atof(argv[8]));

    auto start = std::chrono::high_resolution_clock::now();
    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    IpplTimings::startTimer(mainTimer);
    size_type totalP = std::atol(argv[4]);
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
    double dxFinest = rmax[0] / 1024;  
    const double dt = 0.5 * dxFinest;//size of timestep

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    double Q = -1562.5;
    double Bext = 5.0;
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q);

    P->nr_m = nr;

    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
    static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("domainDecomp");
    static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("Solve");
    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("velocityPush");
    static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("positionPush");

    Vector_t length = rmax - rmin;

    Vector_t mu, sd;

    for (unsigned d = 0; d<Dim; d++) {
        mu[d] = length[d]/2;
    }
    sd[0] = 0.15*length[0];
    sd[1] = 0.05*length[1];
    sd[2] = 0.20*length[2];

    
    P->E_m.initialize(mesh, FL);
    P->rho_m.initialize(mesh, FL);

    bunch_type bunchBuffer(PL);
    //The following update is not needed as the particles are all generated locally
	//IpplTimings::startTimer(updateTimer);
    //PL.update(*P, bunchBuffer);
	//IpplTimings::stopTimer(updateTimer);

    msg << "particles created and initial conditions assigned " << endl;

    P->stype_m = argv[6];
    P->initSolver();
    P->time_m = 0.0;
    P->loadbalancethreshold_m = std::atof(argv[7]);

    unsigned int nstep = 0;
    bool isFirstRepartition;
    
    if (P->balance(totalP, nstep)) {
        msg << "Starting first repartition" << endl;
        IpplTimings::startTimer(domainDecomposition);
        isFirstRepartition = true;
        const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        const int nghost = P->rho_m.getNghost();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto rhoview = P->rho_m.getView();

        Kokkos::parallel_for("Assign initial rho based on PDF",
                              mdrange_type({nghost, nghost, nghost},
                                           {rhoview.extent(0) - nghost,
                                            rhoview.extent(1) - nghost,
                                            rhoview.extent(2) - nghost}),
                              KOKKOS_LAMBDA(const int i,
                                            const int j,
                                            const int k)
                              {
                                //local to global index conversion
                                const size_t ig = i + lDom[0].first() - nghost;
                                const size_t jg = j + lDom[1].first() - nghost;
                                const size_t kg = k + lDom[2].first() - nghost;
                                double x = (ig + 0.5) * hr[0] + origin[0];
                                double y = (jg + 0.5) * hr[1] + origin[1];
                                double z = (kg + 0.5) * hr[2] + origin[2];

                                Vector_t xvec = {x, y, z};

                                rhoview(i, j, k) = PDF(xvec, mu, sd);
                                    
                              });
       
        P->initializeORB(FL, mesh);
        P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
        //IpplTimings::startTimer(particleCreation);
        
        //Compute again the min and max. extents based on the changed region layout
        //for (unsigned d = 0; d <Dim; ++d) {
        //    Nr[d] = CDF(Regions(myRank)[d].max(), mu[d], sd[d]) - 
        //            CDF(Regions(myRank)[d].min(), mu[d], sd[d]);  
        //    Dr[d] = CDF(rmax[d], mu[d], sd[d]) - CDF(rmin[d], mu[d], sd[d]);
        //    minU[d] = CDF(Regions(myRank)[d].min(), mu[d], sd[d]);
        //    maxU[d]   = CDF(Regions(myRank)[d].max(), mu[d], sd[d]);
        //}
        //factor = (Nr[0] * Nr[1] * Nr[2]) / (Dr[0] * Dr[1] * Dr[2]);
        //size_type nlocNew = (size_type)(factor * totalP);
        //size_type TotalParticlesNew = 0;

        //MPI_Allreduce(&nlocNew, &TotalParticlesNew, 1,
        //            MPI_UNSIGNED_LONG, MPI_SUM, Ippl::getComm());

        //int restNew = (int) (totalP - TotalParticlesNew);

        //if ( Ippl::Comm->rank() < restNew )
        //    ++nlocNew;

        //if(nlocNew > P->getLocalNum()) {
        //    //In this case we need to create extra particles
        //    P->create(nlocNew - P->getLocalNum());
        //}
        //else if(nlocNew < P->getLocalNum()) {
        //    
        //    //In this case we need to destroy extra particles
        //    size_type invalidCount = P->getLocalNum() - nlocNew;
        //    
        //    using bool_type = ippl::detail::ViewType<bool, 1>::view_type;

        //    bool_type invalid("invalid", P->getLocalNum());

        //    Kokkos::parallel_for(
        //    "set valid after repartition",
        //    nlocNew,
        //    KOKKOS_LAMBDA(const size_t i) {
        //        invalid(i) = false;
        //    });
        //    Kokkos::fence();

        //    auto pIDs = P->ID.getView();
        //    Kokkos::parallel_for(
        //    "set invalid after repartition",
        //    Kokkos::RangePolicy(nlocNew, P->getLocalNum()),
        //    KOKKOS_LAMBDA(const size_t i) {
        //        invalid(i) = true;
        //        pIDs(i) = -1;
        //    });
        //    Kokkos::fence();
        //    
        //    P->destroy(invalid, invalidCount);
        //}
        //Ippl::Comm->barrier();
        //
        //Kokkos::parallel_for(nlocNew,
        //                     generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
        //                     P->R.getView(), P->P.getView(), rand_pool64, mu, sd, minU, maxU));

        //Kokkos::fence();
        //Ippl::Comm->barrier();
        //IpplTimings::stopTimer(particleCreation);
        IpplTimings::stopTimer(domainDecomposition);
    }

    
    IpplTimings::startTimer(particleCreation);

    typedef ippl::detail::RegionLayout<double, Dim, Mesh_t> RegionLayout_t;
    const RegionLayout_t& RLayout = PL.getRegionLayout();
    const typename RegionLayout_t::host_mirror_type& Regions = RLayout.gethLocalRegions();
    Vector_t Nr, Dr, minU, maxU;
    int myRank = Ippl::Comm->rank();
    for (unsigned d = 0; d <Dim; ++d) {
        Nr[d] = CDF(Regions(myRank)[d].max(), mu[d], sd[d]) - 
                CDF(Regions(myRank)[d].min(), mu[d], sd[d]);  
        Dr[d] = CDF(rmax[d], mu[d], sd[d]) - CDF(rmin[d], mu[d], sd[d]);
        minU[d] = CDF(Regions(myRank)[d].min(), mu[d], sd[d]);
        maxU[d]   = CDF(Regions(myRank)[d].max(), mu[d], sd[d]);
    }

    double factor = (Nr[0] * Nr[1] * Nr[2]) / (Dr[0] * Dr[1] * Dr[2]);
    size_type nloc = (size_type)(factor * totalP);
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
                         P->R.getView(), P->P.getView(), rand_pool64, mu, sd, minU, maxU));

    Kokkos::fence();
    Ippl::Comm->barrier();
    IpplTimings::stopTimer(particleCreation);                                                    
    
    P->q = P->Q_m/totalP;
    isFirstRepartition = false;
    //The following update is not needed as the particles are all generated locally
	//IpplTimings::startTimer(updateTimer);
    //PL.update(*P, bunchBuffer);
	//IpplTimings::stopTimer(updateTimer);
    P->scatterCIC(totalP, 0, hr);

    IpplTimings::startTimer(SolveTimer);
    P->solver_mp->solve();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();

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
        IpplTimings::startTimer(PTimer);
        auto Rview = P->R.getView();
        auto Pview = P->P.getView();
        auto Eview = P->E.getView();
        double V0 = 30*rmax[2];
        Kokkos::parallel_for("Kick1", P->getLocalNum(),
                              KOKKOS_LAMBDA(const size_t j){
            double Eext_x = -(Rview(j)[0] - (rmax[0]/2)) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_y = -(Rview(j)[1] - (rmax[1]/2)) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_z =  (Rview(j)[2] - (rmax[2]/2)) * (V0/(std::pow(rmax[2],2)));

            Pview(j)[0] -= 0.5 * dt * ((Eview(j)[0] + Eext_x) + Pview(j)[1] * Bext);
            Pview(j)[1] -= 0.5 * dt * ((Eview(j)[1] + Eext_y) - Pview(j)[0] * Bext);
            Pview(j)[2] -= 0.5 * dt *  (Eview(j)[2] + Eext_z);
        });
        IpplTimings::stopTimer(PTimer);

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
           P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
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
            double Eext_x = -(R2view(j)[0] - (rmax[0]/2)) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_y = -(R2view(j)[1] - (rmax[1]/2)) * (V0/(2*std::pow(rmax[2],2)));
            double Eext_z =  (R2view(j)[2] - (rmax[2]/2)) * (V0/(std::pow(rmax[2],2)));

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
