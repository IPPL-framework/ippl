// Electrostatic Landau damping test with Particle-in-Fourier schemes
//   Usage:
//     srun ./LandauDampingPIF <nx> <ny> <nz> <Np> <Nt> <dt> <ShapeType> <degree> <tol> --info 5
//     nx       = No. of Fourier modes in the x-direction
//     ny       = No. of Fourier modes in the y-direction
//     nz       = No. of Fourier modes in the z-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     dt       = Time stepsize
//     ShapeType = Shape function type B-spline only for the moment
//     degree = B-spline degree (-1 for delta function)
//     tol = tolerance of NUFFT
//     Example:
//     srun ./LandauDampingPIF 32 32 32 655360 20 0.05 B-spline 1 1e-4 --info 5
//
// Copyright (c) 2022, Sriramkrishnan Muralikrishnan,
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
  
  T k, alpha, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& k_, const T& alpha_, 
           const T& u_) 
  : k(k_), alpha(alpha_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = x  + (alpha  * (std::sin(k * x) / k)) - u;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = 1  + (alpha  * std::cos(k * x));
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

  value_type alpha;

  T k, minU, maxU;

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, 
                  value_type& alpha_, T& k_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        alpha(alpha_), k(k_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u;
    for (unsigned d = 0; d < Dim; ++d) {

        u = rand_gen.drand(minU[d], maxU[d]);
        x(i)[d] = u / (1 + alpha);
        Newton1D<value_type> solver(k[d], alpha, u);
        solver.solve(x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, const double& alpha, const double& k) {
   double cdf = x + (alpha / k) * std::sin(k * x);
   return cdf;
}

KOKKOS_FUNCTION
double PDF(const Vector_t& xvec, const double& alpha, 
             const Vector_t& kw, const unsigned Dim) {
    double pdf = 1.0;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf *= (1.0 + alpha * std::cos(kw[d] * xvec[d]));
    }
    return pdf;
}

const char* TestName = "LandauDampingPIF";

int main(int argc, char *argv[]){
    Ippl ippl(argc, argv);
    
    Inform msg("LandauDampingPIF");
    Inform msg2all("LandauDampingPIF",INFORM_ALL_NODES);

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

    double factor = 1.0/Ippl::Comm->size();
    size_type nloc = (size_type)(factor * totalP);
    size_type Total_particles = 0;

    MPI_Allreduce(&nloc, &Total_particles, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, Ippl::getComm());


    msg << "Landau damping"
        << endl
        << "nt " << nt << " Np= "
        << Total_particles << " Fourier modes = " << nr
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
    Vector_t kw = {0.5, 0.5, 0.5};
    double alpha = 0.05;
    Vector_t rmin(0.0);
    Vector_t rmax = 2 * pi / kw;
    Vector_t length = rmax - rmin;
    double dx = length[0] / nr[0];
    double dy = length[1] / nr[1];
    double dz = length[2] / nr[2];

    Vector_t hr = {dx, dy, dz};
    Vector_t origin = {rmin[0], rmin[1], rmin[2]};

    const bool isAllPeriodic=true;
    Mesh_t mesh(domain, hr, origin);
    FieldLayout_t FL(domain, decomp, isAllPeriodic);
    PLayout_t PL(FL, mesh);

    //Q = -\int\int f dx dv
    double Q = -length[0] * length[1] * length[2];
    P = std::make_unique<bunch_type>(PL,hr,rmin,rmax,decomp,Q,Total_particles);

    P->nr_m = nr;

    P->rho_m.initialize(mesh, FL);
    P->rhoDFT_m.initialize(mesh, FL);
    P->Sk_m.initialize(mesh, FL);

    ////////////////////////////////////////////////////////////
    //Initialize an FFT object for getting rho in real space and 
    //doing charge conservation check
    
    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", false);  
    fftParams.add("use_pencils", true);  
    fftParams.add("use_reorder", false);  
    fftParams.add("use_gpu_aware", true);  
    fftParams.add("comm", ippl::p2p_pl);  
    fftParams.add("r2c_direction", 0);  

    ippl::NDIndex<Dim> domainPIFhalf;

    for(unsigned d = 0; d < Dim; ++d) {
        if(fftParams.template get<int>("r2c_direction") == (int)d)
            domainPIFhalf[d] = ippl::Index(domain[d].length()/2 + 1);
        else
            domainPIFhalf[d] = ippl::Index(domain[d].length());
    }
    

    FieldLayout_t FLPIFhalf(domainPIFhalf, decomp);

    ippl::Vector<double, 3> hDummy = {1.0, 1.0, 1.0};
    ippl::Vector<double, 3> originDummy = {0.0, 0.0, 0.0};
    Mesh_t meshPIFhalf(domainPIFhalf, hDummy, originDummy);

    P->rhoPIFreal_m.initialize(mesh, FL);
    P->rhoPIFhalf_m.initialize(meshPIFhalf, FLPIFhalf);

    P->fft_mp = std::make_shared<FFT_t>(FL, FLPIFhalf, fftParams);
   
    ////////////////////////////////////////////////////////////


    P->time_m = 0.0;

    P->shapetype_m = argv[7]; 
    P->shapedegree_m = std::atoi(argv[8]); 

    IpplTimings::startTimer(particleCreation);

    Vector_t minU, maxU;
    for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin[d], alpha, kw[d]);
        maxU[d]   = CDF(rmax[d], alpha, kw[d]);
    }


    //int rest = (int) (totalP - Total_particles);

    //if ( Ippl::Comm->rank() < rest )
    //    ++nloc;

    P->create(nloc);
    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*Ippl::Comm->rank()));
    Kokkos::parallel_for(nloc,
                         generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                         P->R.getView(), P->P.getView(), rand_pool64, alpha, kw, minU, maxU));

    Kokkos::fence();
    Ippl::Comm->barrier();
    IpplTimings::stopTimer(particleCreation);                                                    
    
    P->q = P->Q_m/Total_particles;
    msg << "particles created and initial conditions assigned " << endl;

    IpplTimings::startTimer(initializeShapeFunctionPIF);
    P->initializeShapeFunctionPIF();
    IpplTimings::stopTimer(initializeShapeFunctionPIF);

    double tol   = std::atof(argv[9]);
    P->initNUFFT(FL,tol);

    P->scatter();

    P->gather();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpLandau();
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
        P->dumpLandau();
        P->dumpEnergy();
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it+1 << " time: " << P->time_m << endl;
    }

    msg << "LandauDamping: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    return 0;
}
