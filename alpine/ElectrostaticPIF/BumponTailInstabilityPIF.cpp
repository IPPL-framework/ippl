// Electrostatic Two-stream/Bump-on-tail instability test with Particle-in-Fourier schemes
//   Usage:
//     srun ./BumponTailInstabilityPIF <nx> <ny> <nz> <Np> <Nt> <dt> <ShapeType> <degree> --info 5
//     nx       = No. of Fourier modes in the x-direction
//     ny       = No. of Fourier modes in the y-direction
//     nz       = No. of Fourier modes in the z-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     dt       = Time stepsize
//     ShapeType = Shape function type B-spline only for the moment
//     degree = B-spline degree (-1 for delta function)
//     Example:
//     srun ./BumponTailInstabilityPIF 32 32 32 655360 20 0.05 B-spline 1 --info 5
//
// Copyright (c) 2023, Sriramkrishnan Muralikrishnan,
// Jülich Supercomputing Centre, Jülich, Germany.
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

#include "ChargedParticlesPIF.hpp"
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
  
  T k, delta, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& k_, const T& delta_, 
           const T& u_) 
  : k(k_), delta(delta_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = x  + (delta  * (std::sin(k * x) / k)) - u;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = 1  + (delta  * std::cos(k * x));
      return Fprime;
  }

  KOKKOS_FUNCTION
  void solve(T& x) {
      int iterations = 0;
      while (iterations < max_iter && std::fabs(f(x)) > tol) {
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

  value_type delta, sigma, muBulk, muBeam;
  size_type nlocBulk; 

  T k, minU, maxU;

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, 
                  value_type& delta_, T& k_, value_type& sigma_, 
                  value_type& muBulk_, value_type& muBeam_, 
                  size_type& nlocBulk_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        delta(delta_), sigma(sigma_), muBulk(muBulk_), muBeam(muBeam_),
        nlocBulk(nlocBulk_), k(k_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    bool isBeam = (i >= nlocBulk);
    
    value_type muZ = (value_type)(((!isBeam) * muBulk) + (isBeam * muBeam));
    
    for (unsigned d = 0; d < Dim-1; ++d) {
        
        x(i)[d] = rand_gen.drand(minU[d], maxU[d]); 
        v(i)[d] = rand_gen.normal(0.0, sigma);
    }
    v(i)[Dim-1] = rand_gen.normal(muZ, sigma);
    
    value_type u = rand_gen.drand(minU[Dim-1], maxU[Dim-1]);
    x(i)[Dim-1] = u / (1 + delta);
    Newton1D<value_type> solver(k[Dim-1], delta, u);
    solver.solve(x(i)[Dim-1]);
    

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, const double& delta, const double& k,
           const unsigned& dim) {

   bool isDimZ = (dim == (Dim-1)); 
   double cdf = x + (double)(isDimZ * ((delta / k) * std::sin(k * x)));
   return cdf;
}

const char* TestName = "TwoStreamInstabilityPIF";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    
    Inform msg(TestName);
    Inform msg2all(TestName,INFORM_ALL_NODES);

    ippl::Vector<int,Dim> nr = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("kick");
    static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("drift");
    static IpplTimings::TimerRef BCTimer = IpplTimings::getTimer("particleBC");
    static IpplTimings::TimerRef initializeShapeFunctionPIF = IpplTimings::getTimer("initializeShapeFunctionPIF");

    IpplTimings::startTimer(mainTimer);

    const size_type totalP = std::atoll(argv[4]);
    const unsigned int nt     = std::atoi(argv[5]);
    const double dt = std::atof(argv[6]);

    msg << TestName 
        << endl
        << "nt " << nt << " Np= "
        << totalP << " Fourier modes = " << nr
        << endl;

    using bunch_type = ChargedParticlesPIF<PLayout_t>;

    std::unique_ptr<bunch_type>  P;

    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i< Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }

    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::SERIAL;
    }

    // create mesh and layout objects for this problem domain
    Vector_t kw;
    double sigma, muBulk, muBeam, epsilon, delta;
    
    if(std::strcmp(TestName,"TwoStreamInstabilityPIF") == 0) {
        // Parameters for two stream instability as in 
        //  https://www.frontiersin.org/articles/10.3389/fphy.2018.00105/full
        kw = {0.5, 0.5, 0.5};
        sigma = 0.1;
        epsilon = 0.5;
        muBulk = -pi / 2.0;
        muBeam = pi / 2.0;
        delta = 0.01;
    }
    else if(std::strcmp(TestName,"BumponTailInstabilityPIF") == 0) {
        kw = {0.21, 0.21, 0.21};
        sigma = 1.0 / std::sqrt(2.0);
        epsilon = 0.1;
        muBulk = 0.0;
        muBeam = 4.0;
        delta = 0.01;
    }
    else {
        //Default value is two stream instability
        kw = {0.5, 0.5, 0.5};
        sigma = 0.1;
        epsilon = 0.5;
        muBulk = -pi / 2.0;
        muBeam = pi / 2.0;
        delta = 0.01;
    }
    
    
    Vector_t rmin(0.0);
    Vector_t rmax = 2 * pi / kw ;
    double dx = rmax[0] / nr[0];
    double dy = rmax[1] / nr[1];
    double dz = rmax[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    //Q = -\int\int f dx dv
    double Q = -rmax[0] * rmax[1] * rmax[2];
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q,totalP);

    P->nr_m = nr;

    P->rho_m.initialize(mesh, FL);
    P->Sk_m.initialize(mesh, FL);

    P->time_m = 0.0;

    P->shapetype_m = argv[7]; 
    P->shapedegree_m = std::atoi(argv[8]); 

    IpplTimings::startTimer(particleCreation);

    //typedef ippl::detail::RegionLayout<double, Dim, Mesh_t> RegionLayout_t;
    //const RegionLayout_t& RLayout = PL.getRegionLayout();
    //const typename RegionLayout_t::host_mirror_type Regions = RLayout.gethLocalRegions();
    Vector_t minU, maxU;
    //int myRank = Ippl::Comm->rank();
    for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin[d], delta, kw[d], d);
        maxU[d]   = CDF(rmax[d], delta, kw[d], d);
    }

    double factorConf = 1.0/Ippl::Comm->size();
    double factorVelBulk = 1.0 - epsilon;
    double factorVelBeam = 1.0 - factorVelBulk;
    size_type nlocBulk = (size_type)(factorConf * factorVelBulk * totalP);
    size_type nlocBeam = (size_type)(factorConf * factorVelBeam * totalP);
    size_type nloc = nlocBulk + nlocBeam;
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
                         P->R.getView(), P->P.getView(), rand_pool64, delta, kw, sigma, muBulk, 
                         muBeam, nlocBulk, minU, maxU));

    Kokkos::fence();
    Ippl::Comm->barrier();
    IpplTimings::stopTimer(particleCreation);                                                    
    
    P->q = P->Q_m/totalP;
    msg << "particles created and initial conditions assigned " << endl;

    IpplTimings::startTimer(initializeShapeFunctionPIF);
    P->initializeShapeFunctionPIF();
    IpplTimings::stopTimer(initializeShapeFunctionPIF);


    P->scatter();

    P->gather();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpBumponTail();
    P->dumpEnergy();
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
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        //drift
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);

        //Apply particle BC
	    IpplTimings::startTimer(BCTimer);
        PL.applyBC(P->R, PL.getRegionLayout().getDomain());
        IpplTimings::stopTimer(BCTimer);

        //scatter the charge onto the underlying grid
        P->scatter();

        // Solve for and gather E field
        P->gather();

        //kick
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);
        P->dumpBumponTail();
        P->dumpEnergy();
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it+1 << " time: " << P->time_m << endl;
    }

    msg << "BumponTailInstability: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}
